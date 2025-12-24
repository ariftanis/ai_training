# Project: Fine-tuning a Turkish Municipality LLM

This project aims to fine-tune a Large Language Model (LLM) to create a specialized chatbot or information retrieval system about the Sancaktepe municipality in Istanbul, Turkey.

## Project Overview

The primary goal is to leverage the documents in the `documents/` directory to train a language model that can answer questions and provide information about the municipality's services, regulations, and general information.

The `plan.md` file outlines a detailed technical plan for this process, which involves:

*   **Model:** Using a base model like `Meta-Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct-v0.3`.
*   **Fine-tuning method:** Employing QLoRA (Quantized Low-Rank Adaptation) for efficient training.
*   **Tooling:** Utilizing the `Unsloth` library for optimized performance on NVIDIA GPUs.

## Dataset

The `documents/` directory contains a collection of Markdown (`.md`) files that serve as the knowledge base for the LLM. These files cover a wide range of topics related to the Sancaktepe municipality, including:

*   **General Information:** Mission, vision, and general details about the municipality.
*   **Services:** Descriptions of services provided by the municipality, such as health, social support, and cultural activities.
*   **Regulations (`Yönetmelikler`):** Official regulations governing the municipality's directorates.
*   **Directorates (`Müdürlükler`):** Information about the various directorates and their functions.
*   **Projects:** Details on completed and ongoing projects.
*   **Facilities:** Information about parks, health centers, and other municipal facilities.

## Training Plan

The `plan.md` file provides a step-by-step guide for the fine-tuning process:

1.  **Environment Setup:** Install Python, CUDA, and the necessary Python libraries (`torch`, `unsloth`, `transformers`, etc.).
2.  **Data Preparation:** Convert the Markdown files into a supervised fine-tuning format (e.g., JSONL with instruction-response pairs).
3.  **Fine-tuning:** Run the training script (`train.py` in `plan.md`) using `Unsloth` and `SFTTrainer`.
4.  **Model Saving and Merging:** Save the fine-tuned model and merge the LoRA adapters for faster inference.
5.  **Testing and Inference:** Test the model's performance and use it for inference.

## Usage

The contents of this directory are intended to be used as a dataset for training a specialized LLM. The `plan.md` file serves as a technical guide for this process. The ultimate goal is to create a model that can act as a knowledgeable assistant for information related to the Sancaktepe municipality.

## How to Run

This project is designed to be run in a containerized environment using Docker and Docker Compose.

### Prerequisites

*   NVIDIA GPU with CUDA drivers installed.
*   Docker and Docker Compose installed on your system.
*   NVIDIA Container Toolkit installed to enable GPU access for Docker.

### Step 1: Prepare the Dataset

First, you need to generate the `dataset.jsonl` file from the documents in the `documents/` directory. Run the following command in your terminal:

```bash
python src/prepare_dataset.py
```

This will create the `dataset.jsonl` file in the root directory.

### Step 2: Train the Model

Next, you need to fine-tune the language model. This is a resource-intensive process that requires a powerful GPU. Run the training script with the following command:

```bash
python src/train.py
```

This script will download the base model and start the fine-tuning process. Upon completion, it will save the fine-tuned model to the `my-finetuned-model/` directory.

**Note:** This step can take a significant amount of time, depending on your hardware. A progress bar will be displayed in your terminal showing the training progress.

### Step 3: Export the Model to GGUF

After training, you can convert your fine-tuned model into a single, highly optimized GGUF file. This format is ideal for running the model with tools like Ollama, LM Studio, or other `llama.cpp`-based applications.

Run the export script:
```bash
python src/export_gguf.py
```

This script automates the following process:
1.  Merges the fine-tuned adapter weights with the base model.
2.  Clones the `llama.cpp` repository from GitHub (if not already present).
3.  Uses the `llama.cpp` tools to convert the merged model into a `sancaktepe-model.gguf` file.

The resulting `.gguf` file is a portable, self-contained version of your model.

### Step 4: Build and Run with Docker Compose

If you wish to run the original Hugging Face model format, you can use the provided Docker setup. Once the model has been trained and the `my-finetuned-model/` directory is present, you can build and run the application using Docker Compose.

To run inference, use the following command:

```bash
docker-compose run qlora-llm python3 src/inference.py "Your prompt here"
```

This command will:
1.  Build the Docker image if it doesn't exist.
2.  Start a container with GPU access.
3.  Run the `inference.py` script with your prompt.
4.  Print the model's response to your terminal.

For example:
```bash
docker-compose run qlora-llm python3 src/inference.py "Sancaktepe belediye başkanı kimdir?"
```
