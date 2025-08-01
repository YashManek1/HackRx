"""
Enhanced Unsloth Fine-tuning Script for Llama3.1:8b
Insurance Q&A Domain Specialization with PDF Processing for HackRx 6.0
Author: YashManek1
Date: 2025-08-01 18:52:44 UTC
"""

import torch
import json
import os
import re
import time
import requests
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Core ML libraries
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
import numpy as np

# PDF processing libraries
try:
    import fitz  # PyMuPDF

    PDF_LIBRARY = "pymupdf"
except ImportError:
    try:
        import pdfplumber

        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_LIBRARY = None

# FastAPI for deployment
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model Configuration - Using your local Ollama model approach
LOCAL_OLLAMA_MODEL = "llama3.1:8b"  # Your local model
MODEL_NAME = "unsloth/llama-3.1-8b-instruct-bnb-4bit"  # Fallback for training
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# Enhanced LoRA Configuration for Better Performance
LORA_CONFIG = {
    "r": 128,  # Higher rank for better domain capture
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": True,  # Better performance
    "loftq_config": None,
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # Reduced for memory
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,  # Increased to maintain effective batch size
    "warmup_ratio": 0.1,
    "num_train_epochs": 5,  # More epochs for custom data
    "max_steps": -1,
    "learning_rate": 1e-4,  # Slightly lower for stability
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "output_dir": "./insurance_qa_model",
    "save_strategy": "steps",
    "save_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "report_to": "none",
    "run_name": f"insurance_qa_llama31_8b_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "logging_dir": "./logs",
    "dataloader_num_workers": 0,
    "remove_unused_columns": False,
}

# Paths and directories
PDF_INPUT_DIR = "./input_pdfs"
OUTPUT_DIR = "./trained_insurance_model"
MERGED_MODEL_DIR = "./trained_insurance_model_merged"
QA_DATASET_PATH = "./generated_qa_dataset.json"
DEPLOYMENT_PORT = 8000
LOCALTUNNEL_PORT = 8000

# =============================================================================
# PDF PROCESSING & Q&A GENERATION
# =============================================================================


def ensure_directories():
    """Create necessary directories"""
    for dir_path in [PDF_INPUT_DIR, OUTPUT_DIR, "./logs"]:
        os.makedirs(dir_path, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF with page information"""
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")

    pages_data = []

    if PDF_LIBRARY == "pymupdf":
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages_data.append(
                    {
                        "page_number": page_num + 1,
                        "text": text.strip(),
                        "source_file": os.path.basename(pdf_path),
                    }
                )
        doc.close()

    elif PDF_LIBRARY == "pdfplumber":
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append(
                        {
                            "page_number": page_num + 1,
                            "text": text.strip(),
                            "source_file": os.path.basename(pdf_path),
                        }
                    )

    else:
        raise ImportError(
            "No PDF library available. Install PyMuPDF: pip install PyMuPDF"
        )

    print(f"✅ Extracted {len(pages_data)} pages from {os.path.basename(pdf_path)}")
    return pages_data


def identify_key_sections(text: str) -> Dict[str, List[str]]:
    """Identify key insurance sections in text"""
    sections = {
        "grace_period": [],
        "waiting_period": [],
        "maternity": [],
        "exclusions": [],
        "coverage": [],
        "claims": [],
        "definitions": [],
        "benefits": [],
        "limits": [],
        "general": [],
    }

    # Split text into sentences
    sentences = re.split(r"[.!?]+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue

        sentence_lower = sentence.lower()

        # Categorize sentences based on keywords
        if any(
            keyword in sentence_lower
            for keyword in ["grace period", "premium payment", "due date"]
        ):
            sections["grace_period"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["waiting period", "pre-existing", "ped"]
        ):
            sections["waiting_period"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["maternity", "pregnancy", "childbirth"]
        ):
            sections["maternity"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["exclusion", "not covered", "excluded"]
        ):
            sections["exclusions"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["coverage", "covered", "benefit", "eligible"]
        ):
            sections["coverage"].append(sentence)
        elif any(
            keyword in sentence_lower for keyword in ["claim", "reimbur", "settlement"]
        ):
            sections["claims"].append(sentence)
        elif any(
            keyword in sentence_lower for keyword in ["define", "means", "shall mean"]
        ):
            sections["definitions"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["sum insured", "limit", "maximum", "minimum"]
        ):
            sections["limits"].append(sentence)
        elif any(
            keyword in sentence_lower
            for keyword in ["health check", "preventive", "wellness"]
        ):
            sections["benefits"].append(sentence)
        else:
            sections["general"].append(sentence)

    return sections


def generate_questions_from_sections(
    sections: Dict[str, List[str]], source_file: str, page_num: int
) -> List[Dict]:
    """Generate Q&A pairs from identified sections"""
    qa_pairs = []

    question_templates = {
        "grace_period": [
            "What is the grace period for premium payment?",
            "How long is the grace period for this policy?",
            "What happens if premium is paid during grace period?",
            "Is there any penalty for paying premium during grace period?",
        ],
        "waiting_period": [
            "What is the waiting period for pre-existing diseases?",
            "When does coverage begin for pre-existing conditions?",
            "Are there exceptions to the waiting period?",
            "How long is the waiting period for specific treatments?",
        ],
        "maternity": [
            "Does this policy cover maternity expenses?",
            "What are the conditions for maternity coverage?",
            "What is the waiting period for maternity benefits?",
            "How many deliveries are covered under this policy?",
        ],
        "exclusions": [
            "What are the major exclusions in this policy?",
            "What medical conditions are not covered?",
            "Are there any treatment exclusions?",
            "What expenses are excluded from coverage?",
        ],
        "coverage": [
            "What medical expenses are covered?",
            "What treatments are included in coverage?",
            "What is the extent of coverage provided?",
            "Are there any specific coverage benefits?",
        ],
        "claims": [
            "What is the claim settlement process?",
            "What documents are required for claims?",
            "What is the claim notification timeline?",
            "How are claims processed under this policy?",
        ],
        "definitions": [
            "How does the policy define key terms?",
            "What is the definition of hospital under this policy?",
            "How are medical terms defined in this policy?",
            "What do specific policy terms mean?",
        ],
        "benefits": [
            "What health check-up benefits are available?",
            "Are there any wellness benefits?",
            "What preventive care is covered?",
            "What additional benefits does this policy offer?",
        ],
        "limits": [
            "What are the policy limits?",
            "Are there any sub-limits on coverage?",
            "What is the maximum coverage amount?",
            "Are there room rent restrictions?",
        ],
    }

    for section_name, sentences in sections.items():
        if not sentences or section_name == "general":
            continue

        # Use the most relevant sentence as answer
        best_sentence = max(sentences, key=len) if sentences else ""
        if len(best_sentence) < 30:  # Skip very short answers
            continue

        # Generate multiple Q&A pairs for this section
        templates = question_templates.get(section_name, [])
        for template in templates[:2]:  # Limit to 2 questions per section
            qa_pairs.append(
                {
                    "instruction": "Answer insurance policy questions with specific details and page references.",
                    "input": template,
                    "output": f"{best_sentence.strip()} [Page {page_num}, {source_file}]",
                    "source_file": source_file,
                    "page_number": page_num,
                    "section": section_name,
                }
            )

    return qa_pairs


def process_pdfs_to_dataset(pdf_directory: str) -> List[Dict]:
    """Process all PDFs in directory and generate training dataset"""
    print("🔍 Scanning for PDF files...")

    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {pdf_directory}")
        print("📥 Please add your insurance PDF files to the input_pdfs directory")
        return []

    print(f"📚 Found {len(pdf_files)} PDF files")

    all_qa_pairs = []

    for pdf_file in pdf_files:
        try:
            # Extract text from PDF
            pages_data = extract_text_from_pdf(str(pdf_file))

            # Process each page
            for page_data in pages_data:
                # Identify key sections
                sections = identify_key_sections(page_data["text"])

                # Generate Q&A pairs
                qa_pairs = generate_questions_from_sections(
                    sections, page_data["source_file"], page_data["page_number"]
                )

                all_qa_pairs.extend(qa_pairs)

            print(
                f"✅ Generated {len([qa for qa in all_qa_pairs if qa['source_file'] == os.path.basename(pdf_file)])} Q&A pairs from {pdf_file.name}"
            )

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            continue

    print(f"🎯 Total Q&A pairs generated: {len(all_qa_pairs)}")

    # Save dataset
    with open(QA_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"💾 Dataset saved to: {QA_DATASET_PATH}")
    return all_qa_pairs


def load_base_insurance_dataset() -> List[Dict]:
    """Load base insurance Q&A dataset"""
    return [
        # Base insurance knowledge
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "What is the grace period for premium payment in health insurance?",
            "output": "The grace period for premium payment is typically thirty (30) days from the premium due date. During this period, the policy remains in force and coverage continues. If premium is not paid within the grace period, the policy may lapse.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "What is the waiting period for pre-existing diseases (PED)?",
            "output": "The waiting period for pre-existing diseases is typically thirty-six (36) months from the policy inception date. Coverage for PED and their direct complications applies only after continuous coverage for 36 months.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "Does health insurance cover maternity expenses?",
            "output": "Yes, most health insurance policies cover maternity expenses including childbirth and lawful medical termination of pregnancy. Conditions typically include: 1) Female insured must have continuous coverage for at least 24 months, 2) Limited to two deliveries or terminations during the policy period, 3) Coverage includes pre and post-natal expenses as specified.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "What is the waiting period for cataract surgery?",
            "output": "The waiting period for cataract surgery is typically two (2) years from policy inception. Both unilateral and bilateral cataract surgeries are subject to this waiting period.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "Are medical expenses for an organ donor covered?",
            "output": "Yes, medical expenses for organ donor hospitalization are typically covered when the organ is donated to an insured person. Coverage applies to harvesting expenses and related medical costs, provided the donation complies with the Transplantation of Human Organs Act, 1994.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "What is the No Claim Discount (NCD) in health insurance?",
            "output": "No Claim Discount (NCD) is typically 5% to 50% on base premium offered on renewal when no claims are made in the preceding year. The discount percentage and maximum accumulation vary by insurer and policy type.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "How does health insurance define a 'Hospital'?",
            "output": "A Hospital is typically defined as an institution with minimum required inpatient beds (10-15 depending on location), qualified nursing staff under medical practitioner supervision available 24/7, fully equipped operation theatre, and maintains daily patient records.",
        },
        {
            "instruction": "Answer insurance policy questions with specific details and page references.",
            "input": "What is the extent of coverage for AYUSH treatments?",
            "output": "Health insurance policies typically cover medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit. Treatment must be taken in a recognized AYUSH Hospital as defined in the policy.",
        },
    ]


def create_combined_dataset() -> List[Dict]:
    """Create combined dataset from PDFs and base knowledge"""
    print("📊 Creating combined training dataset...")

    # Load base dataset
    base_data = load_base_insurance_dataset()
    print(f"📚 Base insurance dataset: {len(base_data)} samples")

    # Load PDF-generated dataset
    pdf_data = []
    if os.path.exists(QA_DATASET_PATH):
        with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
        print(f"📄 PDF-generated dataset: {len(pdf_data)} samples")
    else:
        print("⚠️  No PDF dataset found, processing PDFs...")
        pdf_data = process_pdfs_to_dataset(PDF_INPUT_DIR)

    # Combine datasets
    combined_data = base_data + pdf_data

    # Remove duplicates and filter quality
    seen_questions = set()
    filtered_data = []

    for item in combined_data:
        question = item["input"].lower().strip()
        if question not in seen_questions and len(item["output"]) > 50:
            seen_questions.add(question)
            filtered_data.append(item)

    print(f"🎯 Final dataset size: {len(filtered_data)} samples")
    return filtered_data


def format_training_data(examples: List[Dict]) -> List[Dict]:
    """Format training data into instruction-following format"""
    formatted_data = []

    for example in examples:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

        formatted_data.append({"text": prompt})

    return formatted_data


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_model():
    """Main training function"""
    print("=" * 80)
    print("🚀 ENHANCED INSURANCE Q&A FINE-TUNING FOR LLAMA3.1:8B")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 User: YashManek1")
    print(f"🎯 Target: HackRx 6.0 Competition")
    print("=" * 80)

    # Ensure directories exist
    ensure_directories()

    # Step 1: Create dataset
    print("📊 Creating training dataset...")
    raw_data = create_combined_dataset()

    if len(raw_data) < 10:
        print("⚠️  Warning: Very small dataset. Consider adding more PDFs.")

    formatted_data = format_training_data(raw_data)

    # Split data
    train_size = int(0.9 * len(formatted_data))
    train_data = formatted_data[:train_size]
    eval_data = formatted_data[train_size:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(f"✅ Dataset prepared")
    print(f"📈 Training samples: {len(train_data)}")
    print(f"📊 Evaluation samples: {len(eval_data)}")

    # Step 2: Load Model and Tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            token=None,
        )
        print(f"✅ Loaded {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Step 3: Configure LoRA
    print("\n🔧 Configuring LoRA adaptation...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    print("✅ LoRA configuration applied")

    # Step 4: Configure training
    if TRAINING_CONFIG["max_steps"] == -1:
        steps_per_epoch = len(train_dataset) // (
            TRAINING_CONFIG["per_device_train_batch_size"]
            * TRAINING_CONFIG["gradient_accumulation_steps"]
        )
        TRAINING_CONFIG["max_steps"] = max(
            steps_per_epoch * TRAINING_CONFIG["num_train_epochs"], 100
        )

    training_args = TrainingArguments(**TRAINING_CONFIG)

    # Step 5: Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Step 6: Train
    print("\n🚀 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

    # Step 7: Save models
    print("\n💾 Saving fine-tuned model...")

    # Save LoRA adapters
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ LoRA adapters saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"❌ Failed to save LoRA adapters: {e}")
        return False

    # Save merged model
    try:
        print("🔄 Saving merged model for deployment...")
        model.save_pretrained_merged(
            MERGED_MODEL_DIR,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✅ Merged model saved to: {MERGED_MODEL_DIR}")
    except Exception as e:
        print(f"⚠️  Could not save merged model: {e}")

    # Step 8: Test model
    print("\n🧪 Testing fine-tuned model...")
    test_model(model, tokenizer)

    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    return True


def test_model(model, tokenizer):
    """Test the fine-tuned model"""
    model.eval()

    test_questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?",
    ]

    for i, question in enumerate(test_questions):
        test_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer insurance policy questions with specific details and page references.

### Input:
{question}

### Response:
"""

        try:
            inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("### Response:")[-1].strip()

            print(f"\n❓ Q{i+1}: {question}")
            print(f"✅ A{i+1}: {response}")
        except Exception as e:
            print(f"❌ Test failed for Q{i+1}: {e}")


# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================

# FastAPI app for deployment
app = FastAPI(
    title="Fine-tuned Insurance Q&A API",
    description="Enhanced LLM-powered insurance document Q&A system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
deployed_model = None
deployed_tokenizer = None


class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class QueryResponse(BaseModel):
    answers: List[str]


def load_deployed_model():
    """Load the fine-tuned model for deployment"""
    global deployed_model, deployed_tokenizer

    model_path = MERGED_MODEL_DIR if os.path.exists(MERGED_MODEL_DIR) else OUTPUT_DIR

    if not os.path.exists(model_path):
        print("❌ No trained model found. Please run training first.")
        return False

    try:
        print(f"📦 Loading model from {model_path}...")

        if os.path.exists(MERGED_MODEL_DIR):
            # Load merged model
            deployed_model = AutoModelForCausalLM.from_pretrained(
                MERGED_MODEL_DIR,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )
            deployed_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
        else:
            # Load LoRA model
            deployed_model, deployed_tokenizer = FastLanguageModel.from_pretrained(
                model_name=OUTPUT_DIR,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT,
            )

        deployed_model.eval()
        print("✅ Model loaded successfully for deployment")
        return True

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def generate_answer(question: str) -> str:
    """Generate answer using fine-tuned model"""
    if not deployed_model or not deployed_tokenizer:
        return "Model not loaded"

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer insurance policy questions with specific details and page references.

### Input:
{question}

### Response:
"""

    try:
        inputs = deployed_tokenizer(prompt, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = deployed_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=deployed_tokenizer.eos_token_id,
            )

        response = deployed_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("### Response:")[-1].strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting Fine-tuned Insurance Q&A API...")
    success = load_deployed_model()
    if not success:
        print("⚠️  API started without model. Train model first.")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": deployed_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries(query: QueryRequest):
    """Process document and answer questions"""
    if not deployed_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # For now, we'll use the fine-tuned model to answer questions directly
        # In a full implementation, you'd also process the document URL
        answers = []

        for question in query.questions:
            answer = generate_answer(question)
            answers.append(answer)

        total_time = time.time() - start_time
        print(f"✅ Processed {len(query.questions)} questions in {total_time:.2f}s")

        return QueryResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Fine-tuned Insurance Q&A API",
        "version": "2.0.0",
        "status": "Model loaded" if deployed_model else "Model not loaded",
        "endpoints": ["/health", "/api/v1/hackrx/run"],
    }


def setup_localtunnel():
    """Setup LocalTunnel for public access"""
    try:
        print("🌐 Setting up LocalTunnel...")

        # Install localtunnel if not available
        try:
            subprocess.run(["lt", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📦 Installing LocalTunnel...")
            subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)

        # Start localtunnel in background
        def start_tunnel():
            try:
                result = subprocess.run(
                    ["lt", "--port", str(LOCALTUNNEL_PORT)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.stdout:
                    tunnel_url = result.stdout.strip()
                    print(f"🌍 Public URL: {tunnel_url}")
                    return tunnel_url
            except Exception as e:
                print(f"⚠️  LocalTunnel setup failed: {e}")
                return None

        tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
        tunnel_thread.start()

    except Exception as e:
        print(f"⚠️  LocalTunnel not available: {e}")
        print("💡 Your API will be available locally at http://localhost:8000")


def deploy_to_huggingface():
    """Deploy model to Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo

        print("🤗 Preparing Hugging Face deployment...")

        # Check if model exists
        if not os.path.exists(MERGED_MODEL_DIR):
            print("❌ No merged model found for Hugging Face deployment")
            return False

        # You'll need to set your HF token
        # os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"

        api = HfApi()
        repo_name = (
            f"YashManek1/insurance-qa-llama31-8b-{datetime.now().strftime('%Y%m%d')}"
        )

        print(f"📤 Creating repository: {repo_name}")
        create_repo(repo_name, exist_ok=True)

        print("⬆️  Uploading model files...")
        api.upload_folder(
            folder_path=MERGED_MODEL_DIR,
            repo_id=repo_name,
            commit_message="Fine-tuned Insurance Q&A model for HackRx 6.0",
        )

        print(f"✅ Model deployed to: https://huggingface.co/{repo_name}")
        return True

    except ImportError:
        print("❌ Hugging Face Hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Hugging Face deployment failed: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function"""
    print("🎯 Enhanced Insurance Q&A Fine-tuning & Deployment System")
    print("=" * 60)

    while True:
        print("\n📋 Available Options:")
        print("1. 📄 Process PDFs and generate Q&A dataset")
        print("2. 🏋️  Train fine-tuned model")
        print("3. 🚀 Deploy API locally")
        print("4. 🌐 Deploy with LocalTunnel")
        print("5. 🤗 Deploy to Hugging Face")
        print("6. 🧪 Test model")
        print("7. 📊 View dataset statistics")
        print("8. ❌ Exit")

        choice = input("\n🎯 Enter your choice (1-8): ").strip()

        if choice == "1":
            print("\n📄 Processing PDFs...")
            ensure_directories()
            dataset = process_pdfs_to_dataset(PDF_INPUT_DIR)
            print(f"✅ Generated {len(dataset)} Q&A pairs")

        elif choice == "2":
            print("\n🏋️  Starting model training...")
            success = train_model()
            if success:
                print("🎉 Training completed successfully!")
            else:
                print("❌ Training failed!")

        elif choice == "3":
            print("\n🚀 Starting local API deployment...")
            print("🔧 Loading model...")
            if load_deployed_model():
                print(f"🌟 Starting server on http://localhost:{DEPLOYMENT_PORT}")
                uvicorn.run(app, host="0.0.0.0", port=DEPLOYMENT_PORT, log_level="info")
            else:
                print("❌ Failed to load model for deployment")

        elif choice == "4":
            print("\n🌐 Setting up LocalTunnel deployment...")
            setup_localtunnel()
            if load_deployed_model():
                print(f"🌟 Starting server with LocalTunnel...")
                uvicorn.run(
                    app, host="0.0.0.0", port=LOCALTUNNEL_PORT, log_level="info"
                )
            else:
                print("❌ Failed to load model for deployment")

        elif choice == "5":
            print("\n🤗 Deploying to Hugging Face...")
            success = deploy_to_huggingface()
            if success:
                print("🎉 Successfully deployed to Hugging Face!")
            else:
                print("❌ Hugging Face deployment failed!")

        elif choice == "6":
            print("\n🧪 Testing model...")
            if os.path.exists(OUTPUT_DIR):
                try:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=OUTPUT_DIR,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=None,
                        load_in_4bit=LOAD_IN_4BIT,
                    )
                    test_model(model, tokenizer)
                except Exception as e:
                    print(f"❌ Testing failed: {e}")
            else:
                print("❌ No trained model found. Please train first.")

        elif choice == "7":
            print("\n📊 Dataset Statistics:")
            if os.path.exists(QA_DATASET_PATH):
                with open(QA_DATASET_PATH, "r") as f:
                    data = json.load(f)
                print(f"📄 Total Q&A pairs: {len(data)}")

                # Statistics by source
                sources = defaultdict(int)
                for item in data:
                    sources[item.get("source_file", "unknown")] += 1

                print("📚 By source file:")
                for source, count in sources.items():
                    print(f"   • {source}: {count} pairs")
            else:
                print("❌ No dataset found. Process PDFs first.")

        elif choice == "8":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    # Check system requirements
    if torch.cuda.is_available():
        print(f"🔥 GPU Available: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("⚠️  No GPU available. Training will be slow.")

    # Check PDF library
    if PDF_LIBRARY:
        print(f"📄 PDF processing: {PDF_LIBRARY}")
    else:
        print("❌ No PDF library available. Install PyMuPDF: pip install PyMuPDF")

    main()
