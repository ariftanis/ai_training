# QWEN AI Training Project - QLoRA LLM Fine-tuning for Municipal Information Systems

## Project Overview

This is a specialized AI project focused on creating a custom Large Language Model (LLM) for the Sancaktepe municipality in Istanbul, Turkey. The project uses QLoRA (Quantized Low-Rank Adaptation) fine-tuning to adapt a base model to answer questions about municipal services, regulations, and general information specific to Sancaktepe. The system leverages Turkish municipal documents stored in the `documents/` directory as the knowledge base for training.

The project leverages the Unsloth library for efficient GPU usage and provides a complete pipeline for training, inference, and conversion to GGUF format for broader deployment.

## Architecture & Technologies

- **Base Model**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (configurable via .env)
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Framework**: PyTorch with CUDA support
- **Optimization Library**: Unsloth (provides 2-3x faster training and 70% less VRAM usage)
- **Containerization**: Docker with NVIDIA Container Toolkit
- **Deployment Format**: GGUF (for efficient inference with tools like Ollama or LM Studio)

## Project Structure

```
D:\2025-12-30-PROJECTS\2025-13-30-QLoRA_Ai_training\ai_training\
├── dataset.jsonl              # Generated dataset in JSONL format
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Container definition
├── entrypoint.sh              # Container startup script
├── GEMINI.md                  # Project documentation
├── HowToUseReadMe.md          # Usage instructions
├── QWEN.md                    # Current file (this documentation)
├── .env                       # Environment variables for configuration
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies
├── .idea/                     # IDE configuration
├── documents/                 # Source documents (Turkish municipal documents)
│   ├── agent_prompts.md
│   ├── E_*.md                # Various municipal documents in Turkish
│   ├── test_markdown.md
│   └── url_bank.json
├── output/                    # Output directory for GGUF models
├── llama.cpp/                 # Llama.cpp repository for GGUF conversion
└── src/                       # Source code
    ├── export_gguf.py         # GGUF export and conversion
    ├── inference.py           # Model inference script
    ├── train.py               # Model training script
    └── validate_dataset.py    # Dataset validation script
```

## Key Components

### 1. Model Training (`src/train.py`)
- Uses the prepared JSONL dataset for supervised fine-tuning
- Implements QLoRA with configurable parameters (rank, alpha, dropout)
- Uses 4-bit quantization to minimize VRAM usage
- Trains with configurable parameters (epochs, batch size, learning rate)
- Saves the fine-tuned model to `my-finetuned-model/` directory

### 2. Inference (`src/inference.py`)
- Loads the fine-tuned model and runs inference
- Uses Alpaca format for prompt structuring
- Processes user queries in Turkish about Sancaktepe municipality
- Generates context-aware responses based on training data

### 3. GGUF Export (`src/export_gguf.py`)
- Merges LoRA adapters with the base model
- Converts the model to GGUF format for broader compatibility
- Uses Unsloth's built-in GGUF support when available
- Creates a portable model file for deployment

### 4. Dataset Validation (`src/validate_dataset.py`)
- Validates the dataset for quality and correctness
- Checks for required fields, empty values, duplicates
- Analyzes content lengths and instruction diversity
- Provides sample records for manual review

## Building and Running

### Prerequisites
- NVIDIA GPU with the latest CUDA drivers installed
- Docker and Docker Compose installed on your system
- NVIDIA Container Toolkit for GPU access in containers

### Docker-based Training and Inference
1. **One-time Training and Inference**:
   ```bash
   docker-compose up
   ```

   **What happens next is automatic:**
   
   1. **First-Time Run (Training)**:
      - If you are running this for the first time, the system will detect that no trained model exists.
      - It will automatically prepare the dataset and begin the **fine-tuning process**.
      - You will see logs showing the model download status, followed by a progress bar for the training itself.
      - This is a long, resource-intensive process that can take many hours.
      - Once complete, the newly fine-tuned model will be saved to a `my-finetuned-model/` directory in your project folder.
      - The container will then run inference with a default prompt and create a GGUF export.

   2. **Subsequent Runs (Inference)**:
      - On every subsequent run, the system will detect the `my-finetuned-model/` directory.
      - It will **skip the training process entirely** and immediately run inference using your already-trained model. This will be much faster.

2. **Custom Inference**:
   ```bash
   docker-compose run --rm qlora-llm python3 src/inference.py "Your custom prompt here"
   ```

   For example:
   ```bash
   docker-compose run --rm qlora-llm python3 src/inference.py "Sancaktepe'de hangi parklar var?"
   ```

### Environment Configuration
The `.env` file contains configurable parameters:
- Model selection and sequence length
- LoRA parameters (rank, alpha, dropout)
- Training parameters (epochs, batch size, learning rate)
- Dataset processing settings
- GGUF export configuration
- Quantization preferences for GGUF output

## Development Conventions

- **Language**: Turkish (for queries and documentation)
- **Format**: Alpaca format for instruction-response pairs during training
- **Modeling**: QLoRA fine-tuning approach with 4-bit quantization
- **Containerization**: NVIDIA CUDA base image with specific versioning

## Dependencies

The project relies on these key libraries (from `requirements.txt`):
- `torch`, `torchvision`, `torchaudio` (with CUDA support)
- `unsloth` (optimized for NVIDIA GPUs)
- `trl`, `peft`, `accelerate`, `bitsandbytes` (for fine-tuning)
- `datasets`, `transformers` (for data handling and model operations)
- `tqdm`, `python-dotenv` (for utilities)

## Usage Examples

The model is designed to answer questions about Sancaktepe municipality such as:
- Administrative structure and personnel (names, titles, departments)
- Municipal services and procedures (addresses, phone numbers, required documents)
- Local regulations and policies
- Available facilities and programs
- Historical information and projects

## Deployment Options

1. **Docker Container**: For consistent environment and GPU access
2. **GGUF Format**: For use with Ollama, LM Studio, or other inference engines
3. **Direct Python**: For development and testing purposes

The project is optimized for various NVIDIA GPU configurations and provides efficient fine-tuning capabilities for municipal information systems.