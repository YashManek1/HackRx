import os
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, HttpUrl
import concurrent.futures

from fastAPI.pipeline import (
    process_and_cache_document,
    search_and_answer_question,
    CACHE_DIR,
)

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


# --- Ensure cache directory exists ---
os.makedirs(CACHE_DIR, exist_ok=True)


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
        # Process document in a separate thread to keep FastAPI responsive
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            db_path_future = executor.submit(
                process_and_cache_document, str(query.documents)
            )

            # Wait for document processing to complete
            db_path = db_path_future.result()

        processing_time = time.time() - start_time
        print(f"Document processing completed in {processing_time:.2f} seconds")

        # Process all questions in parallel for speed
        with concurrent.futures.ThreadPoolExecutor() as executor:
            answer_futures = {
                executor.submit(search_and_answer_question, q, db_path): i
                for i, q in enumerate(query.questions)
            }

            # Collect answers in correct order
            answers = [""] * len(query.questions)
            for future in concurrent.futures.as_completed(answer_futures):
                idx = answer_futures[future]
                try:
                    answers[idx] = future.result()
                except Exception as e:
                    answers[idx] = f"Error processing question: {str(e)}"

        total_time = time.time() - start_time
        print(f"Total query processing completed in {total_time:.2f} seconds")

        return QueryResponse(answers=answers)

    except Exception as e:
        print(f"Error in API endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )


# Add to main.py
@app.on_event("startup")
async def startup_event():
    try:
        # Force CPU mode to avoid CUDA/meta tensor issues
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Pre-warm embedding model by loading it once
        from sentence_transformers import SentenceTransformer
        from fastAPI.pipeline import EMBEDDING_MODEL
        
        print("Pre-warming embedding model...")
        global embedding_model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        print("Successfully pre-warmed embedding model")
        
        # Try to pre-warm Ollama
        try:
            import requests
            print("Testing Ollama connection...")
            response = requests.post(
                "http://localhost:11434/api/generate", 
                json={"model": "llama3.1:8b", "prompt": "Hello", "stream": False},
                timeout=5
            )
            print("Successfully connected to Ollama")
        except Exception as ollama_err:
            print(f"Ollama not available: {ollama_err}")
            print("Will use rule-based fallback for answers")
    except Exception as e:
        print(f"Error in startup: {e}")
        print("Continuing with on-demand model loading")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
