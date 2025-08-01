import os
import time
import concurrent.futures
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, HttpUrl
import torch

# Fix the import - use relative import or direct import
try:
    from fastAPI.pipeline import process_questions, EMBEDDING_MODEL, OLLAMA_MODEL
except ImportError:
    # Fallback if pipeline.py is in a different location
    import sys

    sys.path.append(".")
    from fastAPI.pipeline import process_questions, EMBEDDING_MODEL, OLLAMA_MODEL

app = FastAPI(title="LLM-Powered Insurance Document QA API")

# --- Config ---
API_TOKEN = "ac87f477355b48f584ff8cfda844019371e02ec24ea2493df2ba7baa7682e461"


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class QueryResponse(BaseModel):
    answers: List[str]


# --- Auth dependency ---
def verify_token(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return token


# --- Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries(query: QueryRequest, token: str = Depends(verify_token)):
    """
    Process document and answer questions in a single endpoint.

    Args:
        query: Request containing document URL and list of questions

    Returns:
        QueryResponse: List of answers corresponding to questions
    """
    start_time = time.time()

    try:
        # Process document and questions
        answers = process_questions(str(query.documents), query.questions)

        total_time = time.time() - start_time
        print(f"Total processing completed in {total_time:.2f} seconds")

        return QueryResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OLLAMA_MODEL,
        "timestamp": time.time(),
    }


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Pre-warm embedding model
        from sentence_transformers import SentenceTransformer

        print(f"Pre-warming embedding model on {device}...")

        with torch.device(device):
            embedding_model = SentenceTransformer(
                "BAAI/bge-small-en-v1.5", device=device
            )
            embedding_model.eval()

        print(f"Successfully pre-warmed embedding model on {device}")

        # Make model available globally
        try:
            import fastAPI.pipeline

            fastAPI.pipeline.global_embedding_model = embedding_model
            print("Successfully set global embedding model")
        except Exception as e:
            print(f"Warning: Could not set global embedding model: {e}")

        # Test Ollama connection
        try:
            import requests

            print("Testing Ollama connection...")
            response = requests.post(
                f"http://localhost:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": "Hello", "stream": False},
                timeout=5,
            )
            if response.status_code == 200:
                print("Successfully connected to Ollama")
            else:
                print(f"Ollama responded with status {response.status_code}")
        except Exception as ollama_err:
            print(f"Ollama not available: {ollama_err}")
            print("Will use pattern-based fallback for answers")

        # Print system information
        print("=== System Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        print(f"Embedding model: BAAI/bge-small-en-v1.5")
        print(f"LLM model: llama3.1:8b")
        print("=========================")

    except Exception as e:
        print(f"Error in startup: {e}")
        print("Continuing with on-demand model loading")


# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
