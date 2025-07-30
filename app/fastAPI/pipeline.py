import os
import re
import shutil
import requests
import mimetypes
import tempfile
import time
import concurrent.futures
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Generator
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import tiktoken
from urllib.parse import urlparse
import torch

# --- Config ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
MAX_WORKERS = 4  # For parallel processing
MAX_TOKENS_PER_CHUNK = 512  # For token efficiency
CHUNK_OVERLAP = 100  # For sliding window
BATCH_SIZE = 32  # For embedding generation
CACHE_DIR = "document_cache"  # Directory to cache document indexes
NUM_RELEVANT_CHUNKS = 5  # Number of chunks to retrieve per question
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


# 1. Document Download with timeout and retry
def download_file(
    url: str, retries: int = 3, timeout: int = 30
) -> Tuple[str, Optional[str]]:
    """Download file from URL with retry logic"""
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            parsed = urlparse(url)
            suffix = os.path.splitext(parsed.path)[1]  # Only file extension

            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as tmp:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        tmp.write(chunk)
            mime_type, _ = mimetypes.guess_type(temp_path)
            return temp_path, mime_type

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= retries:
                raise RuntimeError(
                    f"Failed to download file after {retries} attempts: {e}"
                )
            time.sleep(1)


# 2. Enhanced Document Parsing
# Using PyMuPDF for faster PDF extraction
def extract_pdf_text_pymupdf(path: str) -> List[str]:
    """Extract text from PDF using PyMuPDF (faster than pdfplumber)"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Please install PyMuPDF: pip install PyMuPDF")

    pages = []
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    return pages


# Using docx2txt for Word docs
def extract_docx_text_docx2txt(path: str) -> List[str]:
    """Extract text from DOCX using docx2txt"""
    try:
        import docx2txt
    except ImportError:
        raise ImportError("Please install docx2txt: pip install docx2txt")

    text = docx2txt.process(path)
    # Split by page breaks (approximation)
    pages = re.split(r"\f", text)
    return [page for page in pages if page.strip()]


# Updated email extraction
def extract_msg_text(path: str) -> List[str]:
    """Extract text from Outlook MSG files"""
    try:
        import extract_msg
    except ImportError:
        raise ImportError("Please install extract-msg: pip install extract-msg")

    msg = extract_msg.Message(path)
    sections = []

    # Include headers and metadata
    headers = []
    if msg.subject:
        headers.append(f"Subject: {msg.subject}")
    if msg.sender:
        headers.append(f"From: {msg.sender}")
    if msg.to:
        headers.append(f"To: {msg.to}")
    if msg.date:
        headers.append(f"Date: {msg.date}")

    if headers:
        sections.append("\n".join(headers))

    # Body text
    if msg.body:
        sections.append(msg.body)

    # Attachments list (names only)
    if msg.attachments:
        att_section = "Attachments:\n" + "\n".join(
            [a.longFilename or a.shortFilename or "Unnamed" for a in msg.attachments]
        )
        sections.append(att_section)

    return ["\n\n".join(sections)]


def extract_pages_by_type(path: str, mime_type: str) -> List[str]:
    """Route to appropriate parser based on file type"""
    ext = os.path.splitext(path)[1].lower()

    if (mime_type and ("pdf" in mime_type)) or ext == ".pdf":
        try:
            return extract_pdf_text_pymupdf(path)
        except Exception as e:
            print(f"PyMuPDF failed: {e}, falling back to pdfplumber")
            return extract_pdf_text_pdfplumber(path)
    elif (mime_type and "word" in mime_type) or ext in {".docx", ".doc"}:
        try:
            return extract_docx_text_docx2txt(path)
        except Exception as e:
            print(f"docx2txt failed: {e}, falling back to python-docx")
            return extract_docx_text_python_docx(path)
    elif (mime_type and "ms-outlook" in mime_type) or ext == ".msg":
        return extract_msg_text(path)
    else:
        raise RuntimeError(f"Unsupported file type: {mime_type} / {ext}")


# Fallback methods in case primary parsers fail
def extract_pdf_text_pdfplumber(path: str) -> List[str]:
    """Fallback PDF text extraction using pdfplumber"""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Please install pdfplumber: pip install pdfplumber")

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
    return pages


def extract_docx_text_python_docx(path: str) -> List[str]:
    """Fallback DOCX text extraction using python-docx"""
    try:
        import docx
    except ImportError:
        raise ImportError("Please install python-docx: pip install python-docx")

    doc = docx.Document(path)
    # Group paragraphs by page breaks (manual heuristic)
    pages = []
    curr = []
    for p in doc.paragraphs:
        if p.text.strip() == "" and curr:
            pages.append("\n".join(curr))
            curr = []
        elif p.text.strip():
            curr.append(p.text.strip())
    if curr:
        pages.append("\n".join(curr))
    return pages


# 3. Improved Text Cleaning
def clean_text(text: str) -> str:
    """Enhanced text cleaning with better handling of special characters and formatting"""
    # Remove common header/footer patterns
    text = re.sub(
        r"Bajaj Allianz General Insurance Co\. Ltd\..*?Issuing Office:",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"UIN-\s*BAJHLIP\d+V\d+\s*", "", text)
    text = re.sub(r"Global Health Care/\s*Policy Wordings/Page \d+", "", text)

    # Fix newlines and spacing issues
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # Limit consecutive newlines

    # Split into lines and remove empty ones
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip() != ""]

    # Rejoin with better handling of paragraph breaks
    output_lines = []
    last_blank = False
    for line in lines:
        if line == "":
            if not last_blank:
                output_lines.append(line)
            last_blank = True
        else:
            output_lines.append(line)
            last_blank = False

    return "\n".join(output_lines).strip()


# 4. Advanced Chunking Strategies
@dataclass
class Chunk:
    """Class for storing document chunks with metadata"""

    text: str
    page: int
    chunk_type: str  # 'sentence', 'paragraph', 'section'
    index_in_page: int
    tokens: int = 0


def count_tokens(text: str) -> int:
    """Count tokens using the tiktoken library"""
    return len(tokenizer.encode(text))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK"""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize

    # Handle potential unicode issues
    if not text:
        return []

    # Improve sentence splitting with better handling of special cases
    # Replace common abbreviations before splitting
    text = re.sub(r"(Mr\.|Mrs\.|Dr\.|Inc\.|Ltd\.|Fig\.|St\.|vs\.)", r"\1<ABBR>", text)

    sentences = sent_tokenize(text)

    # Restore abbreviations
    sentences = [re.sub(r"<ABBR>", ".", s) for s in sentences]

    # Filter out empty or whitespace-only sentences
    return [s for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on double newlines"""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def identify_sections(text: str) -> List[Tuple[str, str]]:
    """Identify sections in text based on common patterns in legal/insurance documents"""
    # Look for section headers
    section_patterns = [
        r"^[A-Z][.)]\s+[A-Z][A-Za-z\s]+$",  # A. SECTION TITLE
        r"^\d+[.)]\s+[A-Z][A-Za-z\s]+$",  # 1. Section Title
        r"^[A-Z][A-Z\s]+:$",  # SECTION TITLE:
        r"^SECTION\s+[A-Z][A-Za-z\s]+$",  # SECTION TITLE
    ]

    # Split by potential section headers
    lines = text.split("\n")
    sections = []
    current_section_title = "Introduction"
    current_content = []

    for line in lines:
        is_section_header = False
        for pattern in section_patterns:
            if re.match(pattern, line.strip()):
                is_section_header = True
                break

        if is_section_header:
            # Save previous section
            if current_content:
                sections.append((current_section_title, "\n".join(current_content)))
                current_content = []
            current_section_title = line.strip()
        else:
            current_content.append(line)

    # Add last section
    if current_content:
        sections.append((current_section_title, "\n".join(current_content)))

    return sections


def create_sliding_window_chunks(
    text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Create chunks using sliding window approach"""
    if not text:
        return []

    tokens = tokenizer.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_tokens = tokens[i : i + max_tokens]
        # Convert back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        # Move window, with overlap
        i += max_tokens - overlap

    return chunks


def generate_chunks(page_text: str, page_num: int) -> List[Chunk]:
    """Generate multiple types of chunks from page text"""
    chunks = []

    # 1. Sentence-level chunks
    sentences = split_into_sentences(page_text)
    for i, sentence in enumerate(sentences):
        token_count = count_tokens(sentence)
        if token_count > 0:  # Skip empty sentences
            chunks.append(
                Chunk(
                    text=sentence,
                    page=page_num,
                    chunk_type="sentence",
                    index_in_page=i,
                    tokens=token_count,
                )
            )

    # 2. Paragraph-level chunks
    paragraphs = split_into_paragraphs(page_text)
    for i, para in enumerate(paragraphs):
        token_count = count_tokens(para)
        if token_count <= MAX_TOKENS_PER_CHUNK and token_count > 0:
            chunks.append(
                Chunk(
                    text=para,
                    page=page_num,
                    chunk_type="paragraph",
                    index_in_page=i,
                    tokens=token_count,
                )
            )
        else:
            # If paragraph is too long, break into smaller chunks
            para_chunks = create_sliding_window_chunks(para, MAX_TOKENS_PER_CHUNK)
            for j, smaller_chunk in enumerate(para_chunks):
                if smaller_chunk.strip():
                    chunks.append(
                        Chunk(
                            text=smaller_chunk,
                            page=page_num,
                            chunk_type="paragraph_part",
                            index_in_page=i * 100 + j,  # Preserve ordering
                            tokens=count_tokens(smaller_chunk),
                        )
                    )

    # 3. Section-level chunks
    sections = identify_sections(page_text)
    for i, (title, content) in enumerate(sections):
        section_text = f"{title}\n{content}"
        token_count = count_tokens(section_text)

        if token_count <= MAX_TOKENS_PER_CHUNK and token_count > 0:
            chunks.append(
                Chunk(
                    text=section_text,
                    page=page_num,
                    chunk_type="section",
                    index_in_page=i,
                    tokens=token_count,
                )
            )
        else:
            # If section is too long, create sliding window chunks
            section_chunks = create_sliding_window_chunks(
                section_text, MAX_TOKENS_PER_CHUNK
            )
            for j, smaller_chunk in enumerate(section_chunks):
                if smaller_chunk.strip():
                    chunks.append(
                        Chunk(
                            text=smaller_chunk,
                            page=page_num,
                            chunk_type="section_part",
                            index_in_page=i * 100 + j,
                            tokens=count_tokens(smaller_chunk),
                        )
                    )

    return chunks


# 5. Optimized Embedding Generation
def get_embeddings_batched(model, texts: List[str], batch_size: int = BATCH_SIZE):
    """Generate embeddings in optimized batches"""
    if not texts:
        return []

    # Process in batches for memory efficiency
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False)

        # Ensure proper formatting of output
        if isinstance(embs, np.ndarray):
            if embs.ndim == 1:
                embs = np.expand_dims(embs, axis=0)
            batch_embs = embs.tolist()
        elif isinstance(embs, list):
            if len(embs) > 0 and isinstance(embs[0], (float, int)):
                batch_embs = [embs]
            else:
                batch_embs = embs
        else:
            raise RuntimeError(
                f"Unexpected output type from model.encode: {type(embs)}"
            )

        all_embeddings.extend(batch_embs)

    return all_embeddings


# 6. Improved FAISS Indexing with incremental updates
class CustomSentenceTransformerEmbeddings(Embeddings):
    """Custom embedding interface for LangChain"""

    def __init__(self, sentence_transformer_model, precomputed_embeddings=None):
        self.model = sentence_transformer_model
        self.precomputed_embeddings = precomputed_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.precomputed_embeddings is not None and len(texts) == len(
            self.precomputed_embeddings
        ):
            return self.precomputed_embeddings
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


def init_faiss_index(embedding_dim: int = 384) -> faiss.IndexFlatIP:
    """Initialize a new FAISS index"""
    return faiss.IndexFlatIP(embedding_dim)


def update_faiss_index(
    db_or_index,
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    faiss_index_path: str,
    model: SentenceTransformer,
):
    """Update existing FAISS index or create a new one"""
    if not texts:
        return db_or_index

    # Convert embeddings to numpy array for FAISS
    if isinstance(db_or_index, FAISS):
        # Update existing LangChain FAISS wrapper
        provider = CustomSentenceTransformerEmbeddings(
            sentence_transformer_model=model, precomputed_embeddings=embeddings
        )
        db = FAISS.from_texts(texts, provider, metadatas=metadatas)

        # Merge with existing index
        db_or_index.merge_from(db)
        return db_or_index
    else:
        # Create new LangChain FAISS wrapper
        provider = CustomSentenceTransformerEmbeddings(
            sentence_transformer_model=model, precomputed_embeddings=embeddings
        )
        db = FAISS.from_texts(texts, provider, metadatas=metadatas)
        return db


def save_faiss_index(db: FAISS, faiss_index_path: str):
    """Save FAISS index to disk"""
    if os.path.exists(faiss_index_path):
        # Move to backup before saving to prevent corruption if interrupted
        backup_path = f"{faiss_index_path}_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(faiss_index_path, backup_path)

    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)

    # Clean up backup if successful
    backup_path = f"{faiss_index_path}_backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)


# UPDATED: Robust FAISS index loading with meta tensor error handling
def load_faiss_index(
    faiss_index_path: str = FAISS_INDEX_PATH, model_name: str = EMBEDDING_MODEL
):
    """Load FAISS index from disk with robust error handling for meta tensor issues"""
    try:
        # Force CPU mode to avoid meta tensor issues
        device = "cpu"
        model = SentenceTransformer(model_name, device=device)
        provider = CustomSentenceTransformerEmbeddings(model)

        if os.path.exists(faiss_index_path):
            try:
                return FAISS.load_local(
                    faiss_index_path, provider, allow_dangerous_deserialization=True
                )
            except RuntimeError as e:
                if "meta tensor" in str(e):
                    print(f"Warning: Meta tensor error when loading index: {e}")
                    # Create a fresh index instead of loading the problematic one
                    if os.path.exists(faiss_index_path):
                        backup_path = f"{faiss_index_path}_problematic"
                        if os.path.exists(backup_path):
                            shutil.rmtree(backup_path)
                        shutil.move(faiss_index_path, backup_path)

                    print("Creating a fresh index")
                    return FAISS.from_texts(
                        ["initialization"], provider, metadatas=[{"empty": True}]
                    )
                raise  # Re-raise if it's not the meta tensor error
        else:
            # Create an empty index if not found
            return FAISS.from_texts(
                ["initialization"], provider, metadatas=[{"empty": True}]
            )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # Return a very basic fallback with a new model instance
        fallback_model = SentenceTransformer(model_name, device="cpu")
        fallback_provider = CustomSentenceTransformerEmbeddings(fallback_model)
        return FAISS.from_texts(
            ["Error loading index"], fallback_provider, metadatas=[{"error": str(e)}]
        )


# 7. Process pages in parallel with ThreadPoolExecutor
def process_page(args):
    """Process a single page (for parallel execution)"""
    page_text, page_num, model, batch_size = args

    # Clean and chunk the text
    cleaned = clean_text(page_text)
    chunks = generate_chunks(cleaned, page_num + 1)  # 1-indexed pages

    if not chunks:
        return None

    # Get just the sentences for compatibility with original code
    sentences = [c.text for c in chunks if c.chunk_type == "sentence"]

    # Extract texts for embedding
    texts = [chunk.text for chunk in chunks]

    # Generate embeddings
    embeddings = get_embeddings_batched(model, texts, batch_size)

    # Create metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append(
            {
                "page": chunk.page,
                "chunk_type": chunk.chunk_type,
                "index_in_page": chunk.index_in_page,
                "token_count": chunk.tokens,
                "text": (
                    chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                ),
            }
        )

    return {
        "page_number": page_num + 1,
        "num_chunks": len(chunks),
        "num_sentences": len(sentences),  # For compatibility with original code
        "chunks": chunks,
        "sentences": sentences,  # Add sentences for compatibility
        "texts": texts,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "token_count": sum(chunk.tokens for chunk in chunks),
    }


# 8. Main processing function with parallel execution and incremental indexing
def process_document_pipeline(
    url: str,
    embedding_model: str = EMBEDDING_MODEL,
    faiss_index_path: str = FAISS_INDEX_PATH,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
) -> Generator[Dict[str, Any], None, None]:
    """Process document with parallel execution and incremental indexing"""
    # First download the document
    path, mime_type = download_file(url)

    try:
        # Load the model once and share across threads
        model = SentenceTransformer(embedding_model)

        # Extract all pages
        pages = extract_pages_by_type(path, mime_type)

        # Initialize FAISS index or load existing
        if os.path.exists(faiss_index_path):
            db = load_faiss_index(faiss_index_path, embedding_model)
        else:
            db = None  # Will be created on first update

        # Process pages in parallel
        page_args = [(pages[i], i, model, batch_size) for i in range(len(pages))]

        # Track token usage
        total_tokens = 0
        total_chunks = 0
        total_sentences = 0

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page processing tasks
            future_to_page = {
                executor.submit(process_page, arg): i for i, arg in enumerate(page_args)
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_idx = future_to_page[future]

                try:
                    result = future.result()

                    if result is None:  # Skip empty pages
                        continue

                    # Update counters
                    total_tokens += result.get("token_count", 0)
                    total_chunks += result.get("num_chunks", 0)
                    total_sentences += result.get("num_sentences", 0)

                    # Update FAISS index incrementally
                    db = update_faiss_index(
                        db,
                        result["texts"],
                        result["embeddings"],
                        result["metadatas"],
                        faiss_index_path,
                        model,
                    )

                    # Save index periodically
                    if page_idx % 10 == 0:
                        save_faiss_index(db, faiss_index_path)

                    # Yield results - ensure backward compatibility
                    yield {
                        "page_number": result["page_number"],
                        "num_sentences": result[
                            "num_sentences"
                        ],  # Keep this for compatibility
                        "sentences": result["sentences"],  # Original API expects this
                        "embeddings": result["embeddings"],
                        "metadatas": result["metadatas"],
                        # New fields
                        "num_chunks": result["num_chunks"],
                        "token_count": result.get("token_count", 0),
                        # Don't include chunks object as it's not serializable
                    }

                except Exception as e:
                    print(f"Error processing page {page_idx + 1}: {e}")
                    # Yield error information
                    yield {
                        "page_number": page_idx + 1,
                        "error": str(e),
                        "num_sentences": 0,
                        "num_chunks": 0,
                        "sentences": [],
                        "embeddings": [],
                        "metadatas": [],
                    }

        # Final save of index
        if db is not None:
            save_faiss_index(db, faiss_index_path)

        # Yield summary stats - FIX: Add page_number field to summary
        yield {
            "page_number": -1,  # Use a special value to indicate summary
            "summary": True,
            "total_pages": len(pages),
            "total_chunks": total_chunks,
            "total_sentences": total_sentences,
            "total_tokens": total_tokens,
            "faiss_index_path": faiss_index_path,
            "num_sentences": 0,  # Add for compatibility
            "sentences": [],  # Add for compatibility
            "embeddings": [],  # Add for compatibility
            "metadatas": [],  # Add for compatibility
        }

    finally:
        # Clean up temp file
        try:
            os.remove(path)
        except Exception:
            pass


# --- NEW FUNCTIONS FOR LLM QUERY PROCESSING ---


def process_and_cache_document(document_url: str) -> str:
    """Process document and return FAISS index path, using cache if available"""
    # Create a unique ID for this document
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = f"{CACHE_DIR}/{url_hash}"

    # Check if already cached
    if os.path.exists(cache_path):
        return cache_path

    # Process document and build index
    print(f"Processing document: {document_url}")
    list(process_document_pipeline(document_url, faiss_index_path=cache_path))

    # Return the cache path
    return cache_path


def build_prompt(context_chunks: List[str], question: str) -> str:
    """Build an optimized prompt for insurance policy Q&A"""

    # Join context chunks, with special emphasis on most relevant
    if len(context_chunks) > 0:
        # Give extra weight to the most relevant chunk
        context = f"Most relevant policy information:\n{context_chunks[0]}\n\n"

        # Add additional context if available
        if len(context_chunks) > 1:
            context += "Additional relevant policy clauses:\n" + "\n\n".join(
                context_chunks[1:]
            )
    else:
        context = "No relevant policy information found."

    # Enhanced prompt for better answers
    prompt = (
        "You are an insurance policy expert answering questions about policy coverage and terms.\n\n"
        "Task: Answer the question using ONLY the provided policy information below. "
        "If the answer is not in the context, say 'Based on the provided policy information, I cannot determine this.'\n\n"
        f"Policy Information:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a direct, concise answer without including phrases like 'Based on the context' or 'According to the document'. "
        "Include specific details like waiting periods, percentages, or conditions when mentioned in the policy information. "
        "Format numbers consistently (use digits for numbers and percentages)."
    )
    return prompt


def query_ollama_llama3(prompt: str, retries: int = 3) -> str:
    """Query Ollama LLM with retry logic for robustness"""
    for attempt in range(retries):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more factual responses
                    "top_k": 40,  # Consider top 40 tokens
                    "top_p": 0.9,  # Sample from top 90% probability mass
                },
            }
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response") or data.get(
                "text", "Error: No response generated"
            )

        except (requests.RequestException, KeyError) as e:
            if attempt == retries - 1:
                return f"Error generating response: {str(e)}"
            time.sleep(1)  # Wait before retry


# UPDATED: Robust search and answer function with better error handling
def search_and_answer_question(question: str, db_path: str) -> str:
    """Process a single question using the FAISS index and LLM with robust error handling"""
    try:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            provider = CustomSentenceTransformerEmbeddings(model)

            if not os.path.exists(db_path) or "_problematic" in db_path:
                raise ValueError("Index unavailable or problematic")
            with torch.no_grad():
                db = FAISS.load_local(
                    db_path, provider, allow_dangerous_deserialization=True
                )

            retrieval_results = db.similarity_search(question, k=NUM_RELEVANT_CHUNKS)
            context_chunks = [doc.page_content for doc in retrieval_results]
            prompt = build_prompt(context_chunks, question)
            try:
                return query_ollama_llama3(prompt)
            except:
                return extract_answer_from_context(question, context_chunks)
        except Exception as e:
            print(f"Search error for '{question}': {e}")
            return "Unable to provide a specific answer due to a technical issue with the search system."
    except Exception as e:
        print(f"Error processing question '{question}': {str(e)}")
        return "Could not process this question due to a technical error."


# ADDED: New fallback function when LLM is unavailable
def extract_answer_from_context(question: str, context_chunks: List[str]) -> str:
    """Extract relevant information from context when LLM is unavailable"""
    if not context_chunks:
        return "No relevant information found."
    question_lower = question.lower()
    keywords = []
    insurance_terms = [
        "grace period",
        "waiting period",
        "pre-existing",
        "maternity",
        "cataract",
        "surgery",
        "organ donor",
        "no claim discount",
        "ncd",
        "preventive",
        "health check",
        "hospital",
        "ayush",
        "sub-limits",
        "room rent",
        "icu",
        "plan a",
    ]
    for term in insurance_terms:
        if term in question_lower:
            keywords.append(term)
    important_words = ["cover", "policy", "limit", "expense", "treatment", "benefit"]
    for word in important_words:
        if word in question_lower:
            keywords.append(word)
    if not keywords:
        stop_words = [
            "what",
            "is",
            "are",
            "the",
            "for",
            "under",
            "this",
            "and",
            "or",
            "in",
            "to",
            "a",
            "an",
        ]
        keywords = [word for word in question_lower.split() if word not in stop_words][
            :3
        ]
    scored_chunks = []
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        score = sum(10 if keyword in chunk_lower else 0 for keyword in keywords)
        sentences = chunk.split(". ")
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            if keyword_count >= 2:
                score += 5 * keyword_count
        if score > 0:
            scored_chunks.append((score, chunk))
    if scored_chunks:
        scored_chunks.sort(reverse=True)
        best_chunks = [chunk for _, chunk in scored_chunks[:2]]
        if "waiting period" in question_lower:
            for chunk in best_chunks:
                if "waiting period" in chunk.lower():
                    for sentence in chunk.split(". "):
                        if (
                            "waiting period" in sentence.lower()
                            and any(str(i) for i in range(10)) in sentence
                        ):
                            return sentence + "."
        if "grace period" in question_lower:
            for chunk in best_chunks:
                if "grace period" in chunk.lower():
                    for sentence in chunk.split(". "):
                        if (
                            "grace period" in sentence.lower()
                            and any(str(i) for i in range(10)) in sentence
                        ):
                            return sentence + "."
        return ". ".join(best_chunks)
    return "Based on the available information, I cannot provide a specific answer to this question."
