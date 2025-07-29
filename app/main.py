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
    Query the FAISS index with a natural language question.
    """
    try:
        db = load_faiss_index()
        docs = db.similarity_search(req.query, k=req.top_k)
        results = []
        for doc in docs:
            results.append({"content": doc.page_content, "metadata": doc.metadata})
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("fastAPI.main:app", host="0.0.0.0", port=8000, reload=True)
