from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from fastAPI.pipeline import (
    process_document_pipeline,
    load_faiss_index,
    FAISS_INDEX_PATH,
)

app = FastAPI(title="Intelligent Doc Ingestion and Embedding API")


class IngestRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    chunk_types: List[str] = ["sentence", "paragraph", "section", "paragraph_part", "section_part"]
    min_token_count: int = 0
    max_token_count: int = 1000
    include_scores: bool = True


@app.post("/ingest")
def ingest_document(req: IngestRequest):
    try:
        # Streaming page-wise processing
        results = []
        for page_result in process_document_pipeline(req.url):
            # You can log or print here for live feedback
            results.append(
                {
                    "page_number": page_result["page_number"],
                    "num_sentences": page_result["num_sentences"],
                    "sample_sentence": (
                        page_result["sentences"][0] if page_result["sentences"] else ""
                    ),
                    "embedding_shape": (  # FIX IS HERE
                        len(
                            page_result["embeddings"][0]
                        )  # Just get the length directly
                        if page_result["embeddings"]
                        else 0
                    ),
                    "metadata_sample": (
                        page_result["metadatas"][0] if page_result["metadatas"] else {}
                    ),
                }
            )
        return {
            "status": "success",
            "faiss_index_path": FAISS_INDEX_PATH,
            "pages_processed": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_faiss(req: QueryRequest):
    """
    Comprehensive query endpoint with intelligent chunk retrieval and filtering.
    Supports basic and advanced querying with multi-factor ranking.
    """
    try:
        db = load_faiss_index()
        
        # Get more candidates initially for better ranking and filtering
        initial_k = min(max(req.top_k * 5, 50), 200)  # Get 5x more candidates, capped at 200
        
        # Perform similarity search with scores
        docs_with_scores = db.similarity_search_with_score(req.query, k=initial_k)
        
        if not docs_with_scores:
            return {
                "query": req.query,
                "filters_applied": {
                    "chunk_types": req.chunk_types,
                    "min_token_count": req.min_token_count,
                    "max_token_count": req.max_token_count
                },
                "total_candidates_found": 0,
                "filtered_candidates": 0,
                "returned_results": 0,
                "matches": []
            }
        
        # Enhanced ranking and filtering
        ranked_results = []
        
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            chunk_type = metadata.get("chunk_type", "sentence")
            token_count = metadata.get("token_count", 0)
            
            # Apply filters
            if (chunk_type in req.chunk_types and 
                req.min_token_count <= token_count <= req.max_token_count):
                
                # Calculate enhanced relevance score
                relevance_score = score  # Base similarity score
                
                # Boost score based on chunk type (sections and paragraphs are often more informative)
                if chunk_type == "section":
                    relevance_score *= 1.3
                elif chunk_type == "paragraph":
                    relevance_score *= 1.2
                elif chunk_type == "section_part":
                    relevance_score *= 1.1
                elif chunk_type == "paragraph_part":
                    relevance_score *= 1.05
                
                # Boost score based on token count (longer chunks often have more context)
                if token_count > 200:
                    relevance_score *= 1.15
                elif token_count > 100:
                    relevance_score *= 1.1
                elif token_count > 50:
                    relevance_score *= 1.05
                
                # Text length factor for quality assessment
                content_length = len(doc.page_content)
                length_score = min(content_length / 500, 1.0)  # Normalize to max 1.0
                
                # Query term overlap boost (simple keyword matching)
                query_terms = set(req.query.lower().split())
                content_terms = set(doc.page_content.lower().split())
                overlap_ratio = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
                overlap_boost = 1.0 + (overlap_ratio * 0.2)  # Up to 20% boost
                
                # Final combined score
                final_score = relevance_score * overlap_boost * (0.7 + 0.3 * length_score)
                
                result = {
                    "content": doc.page_content,
                    "metadata": metadata,
                    "chunk_type": chunk_type,
                    "page_number": metadata.get("page", "unknown"),
                    "token_count": token_count,
                    "content_length": content_length,
                    "query_overlap_ratio": float(overlap_ratio)
                }
                
                if req.include_scores:
                    result["similarity_score"] = float(score)
                    result["relevance_score"] = float(final_score)
                
                ranked_results.append(result)
        
        # Sort by enhanced relevance score (higher is better for FAISS cosine similarity)
        if req.include_scores and ranked_results:
            ranked_results.sort(key=lambda x: x.get("relevance_score", x.get("similarity_score", 0)), reverse=True)
        
        # Return top K results
        final_results = ranked_results[:req.top_k]
        
        # Add ranking information
        for i, result in enumerate(final_results):
            result["rank"] = i + 1
        
        return {
            "query": req.query,
            "filters_applied": {
                "chunk_types": req.chunk_types,
                "min_token_count": req.min_token_count,
                "max_token_count": req.max_token_count
            },
            "total_candidates_found": len(docs_with_scores),
            "filtered_candidates": len(ranked_results),
            "returned_results": len(final_results),
            "matches": final_results
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="FAISS index not found. Please ingest a document first using the /ingest endpoint."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("fastAPI.main:app", host="0.0.0.0", port=8000, reload=True)
