# Step-by-Step TODO Plan: QLoRA LLM Fine-Tuning and Dockerization

This plan outlines the process of preparing a dataset, fine-tuning a Large Language Model (LLM), converting it to GGUF for efficient inference, and packaging it into a Docker container. For the fine-tuning process, the **Llama 3.1 8B Instruct model** will be used, specifically the `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` variant due to its efficiency with QLoRA.

## System Environment

*   **Operating System:** Ubuntu 24.04 LTS
*   **CUDA Support:** Fully enabled and supported by the system drivers. This allows for GPU acceleration during model training and inference.

### Phase 1: Project Setup and Data Preparation

1.  **Initialize Project Structure:**
    *   Create a `src` directory for Python scripts.

2.  **Set Up Python Environment:**
    *   Create a Python virtual environment (e.g., `python -m venv .venv`).
    *   Activate the virtual environment.

3.  **Install Dependencies:**
    *   Install all necessary libraries using pip:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        pip install "unsloth[cu130] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps "trl" "peft" "accelerate" "bitsandbytes"
        pip install datasets transformers
        ```

4.  **Create Data Preparation Script (`src/prepare_dataset.py`):**
    *   This script reads all `.md` and `.json` files from the `documents/` directory, converts them into a structured JSONL format using the Alpaca standard, and saves the result to `dataset.jsonl`.

### Phase 2: Model Training

5.  **Create Training Script (`src/train.py`):**
    *   The script loads the `dataset.jsonl`, fine-tunes the base model using QLoRA, and saves the resulting adapter weights to the `./my-finetuned-model` directory.

6.  **Run the Training Process:**
    *   Execute the training script: `python src/train.py`. This is a long-running, resource-intensive process. You will see logs indicating the model download progress, followed by a `tqdm` progress bar for the training steps.
    *   Upon completion, verify that the `my-finetuned-model/` directory has been created.

### Phase 3: Model Export and GGUF Conversion

7.  **Create GGUF Export Script (`src/export_gguf.py`):**
    *   This script will first merge the LoRA adapters from `my-finetuned-model/` into the base model, saving the result to a new directory (`my-finetuned-merged/`).
    *   It will then automatically clone the `llama.cpp` repository (if not already present), install its dependencies, and use its conversion tools to create a GGUF file (e.g., `sancaktepe-model.gguf`).

8.  **Run the GGUF Export Process:**
    *   Execute the export script: `python src/export_gguf.py`.
    *   Verify that the `.gguf` model file is created in the root directory. This single file is now a portable, high-performance version of your fine-tuned model.

### Phase 4: Dockerization and Deployment

9.  **Create `requirements.txt`:**
    *   Generate a `requirements.txt` file with all the project dependencies: `pip freeze > requirements.txt`.

10. **Create Inference Script (`src/inference.py`):**
    *   This script loads the fine-tuned model (the Hugging Face version, not the GGUF) and runs inference. It serves as a way to test the model outside of a GGUF-based environment.

11. **Write the `Dockerfile`:**
    *   Create a `Dockerfile` that uses a base image like `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`, copies the necessary scripts and the Hugging Face model, and sets up the environment to run inference.

12. **Create `docker-compose.yml` file:**
    *   Create a `docker-compose.yml` file to simplify container management and ensure GPU access for running the Hugging Face model format.

13. **Build and Test with Docker Compose:**
    *   Build and run the service using: `docker-compose up --build`.

### Phase 5: Documentation

14. **Update `GEMINI.md`:**
    *   Add sections to `GEMINI.md` explaining the GGUF conversion process and how to use the final GGUF model with tools like Ollama or LM Studio.
