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
        initial_k = min(max(req.top_k * 10, 100), 500)  # Get 10x more candidates, capped at 500
        
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
        
        # Enhanced ranking and filtering with deduplication
        ranked_results = []
        seen_content = set()  # To avoid duplicate content
        
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            chunk_type = metadata.get("chunk_type", "sentence")
            token_count = metadata.get("token_count", 0)
            content = doc.page_content.strip()
            
            # Skip duplicates based on content hash
            content_hash = hash(content[:200])  # Use first 200 chars for duplicate detection
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Apply filters
            if (chunk_type in req.chunk_types and 
                req.min_token_count <= token_count <= req.max_token_count and
                len(content) > 20):  # Skip very short content
                
                # Calculate enhanced relevance score
                relevance_score = score  # Base similarity score
                
                # Boost score based on chunk type (sections and paragraphs are often more informative)
                if chunk_type == "section":
                    relevance_score *= 1.4
                elif chunk_type == "paragraph":
                    relevance_score *= 1.3
                elif chunk_type == "section_part":
                    relevance_score *= 1.2
                elif chunk_type == "paragraph_part":
                    relevance_score *= 1.1
                
                # Boost score based on token count (longer chunks often have more context)
                if token_count > 200:
                    relevance_score *= 1.2
                elif token_count > 100:
                    relevance_score *= 1.15
                elif token_count > 50:
                    relevance_score *= 1.1
                
                # Enhanced query term overlap boost with better matching
                query_terms = set(req.query.lower().replace("?", "").replace(".", "").split())
                content_terms = set(content.lower().replace("?", "").replace(".", "").split())
                
                # Calculate different types of matches
                exact_matches = len(query_terms.intersection(content_terms))
                partial_matches = 0
                
                # Check for partial word matches (stemming-like)
                for q_term in query_terms:
                    if len(q_term) > 3:  # Only for longer words
                        for c_term in content_terms:
                            if q_term in c_term or c_term in q_term:
                                partial_matches += 0.5
                
                total_matches = exact_matches + partial_matches
                overlap_ratio = total_matches / len(query_terms) if query_terms else 0
                
                # Context-aware matching - check if the context makes sense
                context_relevance = 1.0
                
                # For financial queries, look for financial context
                financial_terms = {"profit", "investment", "rate", "percentage", "yield", "interest", "dividend", "roi", "performance", "gain"}
                travel_terms = {"trip", "journey", "repatriation", "travel", "flight", "ticket", "accommodation"}
                insurance_terms = {"policy", "coverage", "claim", "premium", "benefit", "refund"}
                
                query_lower = req.query.lower()
                content_lower = content.lower()
                
                # If query seems financial but content is about travel/insurance, penalize heavily
                if any(word in query_lower for word in ["return", "annual", "profit", "investment"]):
                    if any(word in content_lower for word in travel_terms):
                        context_relevance *= 0.3  # Heavy penalty for travel context
                    elif any(word in content_lower for word in insurance_terms) and not any(word in content_lower for word in financial_terms):
                        context_relevance *= 0.4  # Penalty for pure insurance context
                    elif any(word in content_lower for word in financial_terms):
                        context_relevance *= 1.2  # Boost for financial context
                
                # More conservative boost for matches
                if overlap_ratio > 0.7 and context_relevance > 0.8:
                    overlap_boost = 1.0 + (overlap_ratio * 0.4)  # Up to 40% boost for good context
                elif overlap_ratio > 0.5 and context_relevance > 0.6:
                    overlap_boost = 1.0 + (overlap_ratio * 0.2)  # Up to 20% boost
                elif overlap_ratio > 0.2:
                    overlap_boost = 1.0 + (overlap_ratio * 0.1)  # Up to 10% boost
                else:
                    overlap_boost = 1.0
                
                # Text length factor for quality assessment
                content_length = len(content)
                if content_length > 1000:
                    length_score = 0.9  # Penalize very long chunks slightly
                elif content_length > 300:
                    length_score = 1.0
                elif content_length > 100:
                    length_score = 0.95
                else:
                    length_score = 0.85  # Penalize very short chunks
                
                # Penalize results that are too generic or repetitive
                generic_penalty = 1.0
                generic_words = {"the", "and", "or", "in", "to", "of", "for", "with", "by", "from", "this", "that", "will", "shall"}
                content_words = content_terms - generic_words
                if len(content_words) < 5:  # Too few meaningful words
                    generic_penalty = 0.8
                
                # Final combined score with context awareness
                final_score = relevance_score * overlap_boost * length_score * generic_penalty * context_relevance
                
                result = {
                    "content": content,
                    "metadata": metadata,
                    "chunk_type": chunk_type,
                    "page_number": metadata.get("page", "unknown"),
                    "token_count": token_count,
                    "content_length": content_length,
                    "query_overlap_ratio": float(overlap_ratio),
                    "exact_matches": exact_matches,
                    "partial_matches": round(partial_matches, 1),
                    "context_relevance": float(context_relevance)
                }
                
                if req.include_scores:
                    result["similarity_score"] = float(score)
                    result["relevance_score"] = float(final_score)
                    result["overlap_boost"] = float(overlap_boost)
                
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
            "deduplication_applied": True,
            "matches": final_results
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="FAISS index not found. Please ingest a document first using the /ingest endpoint."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/debug")
def debug_query(req: QueryRequest):
    """
    Debug endpoint to understand query processing and scoring details.
    """
    try:
        db = load_faiss_index()
        
        # Get fewer candidates for detailed analysis
        initial_k = 20
        docs_with_scores = db.similarity_search_with_score(req.query, k=initial_k)
        
        if not docs_with_scores:
            return {
                "query": req.query,
                "message": "No candidates found",
                "debug_info": []
            }
        
        debug_results = []
        query_terms = set(req.query.lower().replace("?", "").replace(".", "").split())
        
        for i, (doc, score) in enumerate(docs_with_scores[:10]):  # Analyze first 10
            metadata = doc.metadata
            content = doc.page_content.strip()
            content_terms = set(content.lower().replace("?", "").replace(".", "").split())
            
            # Calculate matches
            exact_matches = query_terms.intersection(content_terms)
            overlap_ratio = len(exact_matches) / len(query_terms) if query_terms else 0
            
            debug_results.append({
                "rank": i + 1,
                "similarity_score": float(score),
                "page": metadata.get("page", "unknown"),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "token_count": metadata.get("token_count", 0),
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "query_terms": list(query_terms),
                "exact_matches": list(exact_matches),
                "overlap_ratio": float(overlap_ratio),
                "content_length": len(content)
            })
        
        return {
            "query": req.query,
            "total_candidates_analyzed": len(docs_with_scores),
            "debug_results": debug_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/analyze")
def analyze_query_relevance(req: QueryRequest):
    """
    Analyze why specific results are being returned and their true relevance.
    """
    try:
        db = load_faiss_index()
        
        # Get candidates for analysis
        initial_k = 20
        docs_with_scores = db.similarity_search_with_score(req.query, k=initial_k)
        
        if not docs_with_scores:
            return {
                "query": req.query,
                "message": "No candidates found",
                "analysis": []
            }
        
        analysis_results = []
        query_terms = set(req.query.lower().replace("?", "").replace(".", "").split())
        
        # Define context categories
        financial_terms = {"profit", "investment", "rate", "percentage", "yield", "interest", "dividend", "roi", "performance", "gain", "return", "annual"}
        travel_terms = {"trip", "journey", "repatriation", "travel", "flight", "ticket", "accommodation", "return"}
        insurance_terms = {"policy", "coverage", "claim", "premium", "benefit", "refund", "withdrawal"}
        
        for i, (doc, score) in enumerate(docs_with_scores[:10]):
            metadata = doc.metadata
            content = doc.page_content.strip()
            content_terms = set(content.lower().replace("?", "").replace(".", "").split())
            
            # Calculate matches
            exact_matches = query_terms.intersection(content_terms)
            overlap_ratio = len(exact_matches) / len(query_terms) if query_terms else 0
            
            # Analyze context
            content_lower = content.lower()
            financial_context = sum(1 for term in financial_terms if term in content_lower)
            travel_context = sum(1 for term in travel_terms if term in content_lower)
            insurance_context = sum(1 for term in insurance_terms if term in content_lower)
            
            # Determine primary context
            contexts = {
                "financial": financial_context,
                "travel": travel_context, 
                "insurance": insurance_context
            }
            primary_context = max(contexts, key=contexts.get)
            
            # Calculate true relevance based on query intent
            query_lower = req.query.lower()
            likely_financial_query = any(word in query_lower for word in ["return", "annual", "profit", "investment", "yield"])
            
            if likely_financial_query:
                if primary_context == "financial":
                    true_relevance = "HIGH"
                elif primary_context == "travel" or primary_context == "insurance":
                    true_relevance = "LOW"
                else:
                    true_relevance = "MEDIUM"
            else:
                true_relevance = "MEDIUM"
            
            analysis_results.append({
                "rank": i + 1,
                "similarity_score": float(score),
                "page": metadata.get("page", "unknown"),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "content_preview": content[:150] + "..." if len(content) > 150 else content,
                "exact_matches": list(exact_matches),
                "overlap_ratio": float(overlap_ratio),
                "context_analysis": {
                    "primary_context": primary_context,
                    "financial_terms": financial_context,
                    "travel_terms": travel_context,
                    "insurance_terms": insurance_context
                },
                "true_relevance": true_relevance,
                "relevance_explanation": f"Content is primarily about {primary_context}. Query seems {'financial' if likely_financial_query else 'general'}."
            })
        
        return {
            "query": req.query,
            "query_analysis": {
                "query_terms": list(query_terms),
                "likely_intent": "financial" if likely_financial_query else "general",
                "financial_keywords": [word for word in query_terms if word in financial_terms]
            },
            "total_candidates_analyzed": len(docs_with_scores),
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("fastAPI.main:app", host="0.0.0.0", port=8000, reload=True)
