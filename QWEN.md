# QWEN AI Training Project - QLoRA LLM Fine-tuning for Sancaktepe Municipality

## Project Overview

This is a specialized AI project focused on creating a custom Large Language Model (LLM) for the Sancaktepe municipality in Istanbul, Turkey. The project uses QLoRA (Quantized Low-Rank Adaptation) fine-tuning to adapt a base model (Meta-Llama-3.1-8B-Instruct) to answer questions about municipal services, regulations, and general information specific to Sancaktepe.

The project leverages Turkish municipal documents stored in the `documents/` directory as the knowledge base for training. It utilizes the Unsloth library for efficient GPU usage and provides a complete pipeline for training, inference, and conversion to GGUF format for broader deployment.

## Architecture & Technologies

- **Base Model**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Framework**: PyTorch with CUDA support
- **Optimization Library**: Unsloth (provides 2-3x faster training and 70% less VRAM usage)
- **Containerization**: Docker with NVIDIA Container Toolkit
- **Deployment Format**: GGUF (for efficient inference with tools like Ollama or LM Studio)

## Project Structure

```
D:\ai_training\
├── dataset.jsonl              # Generated dataset in JSONL format
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Container definition
├── entrypoint.sh              # Container startup script
├── GEMINI.md                  # Project documentation
├── plan.md                    # Original planning document
├── QWEN.md                    # Current file (this documentation)
├── requirements.txt           # Python dependencies
├── step_by_step_plan.md       # Step-by-step implementation plan
├── .idea/                     # IDE configuration
├── documents/                 # Source documents (Turkish municipal documents)
│   ├── agent_prompts.md
│   ├── E_*.md                # Various municipal documents in Turkish
│   ├── test_markdown.md
│   └── url_bank.json
└── src/                       # Source code
    ├── export_gguf.py         # GGUF export and conversion
    ├── inference.py           # Model inference script
    ├── prepare_dataset.py     # Dataset preparation from documents
    └── train.py               # QLoRA model training script
```

## Key Components

### 1. Dataset Preparation (`src/prepare_dataset.py`)
- Reads all `.md` and `.json` files from the `documents/` directory
- Converts them into Alpaca format JSONL records
- Handles markdown files by extracting titles and content
- Special handling for `url_bank.json` and generic JSON files
- Creates a structured dataset for training

### 2. Model Training (`src/train.py`)
- Uses the prepared JSONL dataset for supervised fine-tuning
- Implements QLoRA with rank=64 and alpha=16
- Uses 4-bit quantization to minimize VRAM usage
- Trains for 3 epochs with batch size adjustments for stability
- Saves the fine-tuned model to `my-finetuned-model/` directory

### 3. Inference (`src/inference.py`)
- Loads the fine-tuned model and runs inference
- Uses Alpaca format for prompt structuring
- Processes user queries in Turkish about Sancaktepe municipality
- Generates context-aware responses based on training data

### 4. GGUF Export (`src/export_gguf.py`)
- Merges LoRA adapters with the base model
- Converts the model to GGUF format for broader compatibility
- Uses llama.cpp for conversion and optimization
- Creates a portable model file for deployment

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA support (tested with RTX 5090, 32GB VRAM)
- Docker and Docker Compose
- NVIDIA Container Toolkit for GPU access in containers
- CUDA drivers installed on the host system

### Docker-based Training and Inference
1. **One-time Training and Inference**:
   ```bash
   docker-compose up
   ```
   - On first run: Automatically prepares dataset, trains model, and runs inference
   - On subsequent runs: Skips training and runs inference directly

2. **Custom Inference**:
   ```bash
   docker-compose run --rm qlora-llm python3 src/inference.py "Sancaktepe'de hangi parklar var?"
   ```

### Direct Python Usage (Alternative)
1. **Prepare dataset**:
   ```bash
   python src/prepare_dataset.py
   ```

2. **Train model**:
   ```bash
   python src/train.py
   ```

3. **Run inference**:
   ```bash
   python src/inference.py "Your question here"
   ```

4. **Export to GGUF** (after training):
   ```bash
   python src/export_gguf.py
   ```

## Development Conventions

- **Language**: Turkish (for queries and documentation)
- **Format**: Alpaca format for instruction-response pairs during training
- **Modeling**: QLoRA fine-tuning approach with 4-bit quantization
- **Containerization**: NVIDIA CUDA base image with specific versioning
- **File Formats**: Markdown and JSON documents used as knowledge source

## Dependencies

The project relies on these key libraries (from `requirements.txt`):
- `torch`, `torchvision`, `torchaudio` (with CUDA support)
- `unsloth` (optimized for NVIDIA GPUs)
- `trl`, `peft`, `accelerate`, `bitsandbytes` (for fine-tuning)
- `datasets`, `transformers` (for data handling and model operations)

## Usage Examples

The model is designed to answer questions about Sancaktepe municipality such as:
- Administrative structure and personnel
- Municipal services and procedures
- Local regulations and policies
- Available facilities and programs
- Historical information and projects

## Deployment Options

1. **Docker Container**: For consistent environment and GPU access
2. **GGUF Format**: For use with Ollama, LM Studio, or other inference engines
3. **Direct Python**: For development and testing purposes

The project is optimized for the RTX 5090 with 32GB VRAM but can be adapted for other NVIDIA GPU configurations with sufficient memory.