# Insurance Document Q&A API (HackRx 6.0)

This repository contains a FastAPI-based backend and fine-tuning scripts for an LLM-powered insurance document Q&A system. It supports PDF ingestion, domain-specific Q&A generation, and deployment via API.

## Features

- **Document Ingestion:** Upload insurance policy PDFs and extract key sections.
- **Q&A Generation:** Automatically generate insurance Q&A pairs for fine-tuning.
- **LLM Fine-tuning:** Fine-tune Llama3.1:8b (Unsloth/LoRA) for insurance Q&A.
- **API Deployment:** Serve answers via FastAPI endpoints.
- **Local & Public Access:** Deploy locally or via LocalTunnel.
- **Hugging Face Integration:** Optionally deploy model to Hugging Face Hub.

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/HackRx.git
cd HackRx
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment.

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# Or
source venv/bin/activate  # On Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download/Prepare Insurance PDFs

Place your insurance policy PDFs in the `input_pdfs` directory.

### 4. Run Fine-tuning Script

```sh
python app/finetuning.py
```

Follow the interactive menu to process PDFs, train the model, and deploy the API.

### 5. Start the API Server

After training, you can start the API server:

```sh
python app/finetuning.py
```

Choose the deployment option from the menu.

Or, to run the FastAPI server directly:

```sh
python app/main.py
```

### 6. API Usage

- **POST /api/v1/hackrx/run**  
  Request:

  ```json
  {
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period?",
      "Does this policy cover maternity?"
    ]
  }
  ```

  Response:

  ```json
  {
    "answers": [
      "The grace period is thirty (30) days...",
      "Yes, maternity expenses are covered..."
    ]
  }
  ```

- **GET /health**  
  Returns system and model status.

## Notes

- GPU recommended for training and inference.
- Ollama must be running locally for LLM inference (`llama3.1:8b`).
- For PDF support, ensure `PyMuPDF` or `pdfplumber` is installed.
- For Hugging Face deployment, set your `HUGGINGFACE_HUB_TOKEN` as an environment variable.

## Directory Structure

```
app/
  main.py           # FastAPI server
  finetuning.py     # Fine-tuning and deployment script
  fastAPI/
    pipeline.py     # Document processing and Q&A logic
input_pdfs/         # Place your insurance PDFs here
trained_insurance_model/      # Output model directory
trained_insurance_model_merged/ # Merged model for deployment
generated_qa_dataset.json     # Generated Q&A dataset
requirements.txt
README.md
```

## License

MIT

---

\*\*For any issues or contributions, please open an issue or
