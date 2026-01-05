# How to Use This LLM Fine-Tuning Project

## 1. Project Overview

This project provides a complete environment for fine-tuning a Large Language Model (LLM) for a specific task using your own data. It is built using Docker and `unsloth` for efficient, quantized fine-tuning (QLoRA) on consumer-grade NVIDIA GPUs.

The primary use case is to train a specialized chatbot or information retrieval system. The project is currently configured to train a model about the Sancaktepe municipality in Istanbul, Turkey.

## 2. System Requirements

### Hardware
*   **NVIDIA GPU:** A CUDA-enabled NVIDIA GPU is required. The fine-tuning process is resource-intensive, and a GPU with at least 8GB of VRAM is recommended for the default models.

### Software
*   **Docker:** The project is fully containerized, so you will need Docker installed.
*   **Docker Compose:** Used to orchestrate the services. It is usually included with Docker Desktop.
*   **NVIDIA Container Toolkit:** This is required to allow Docker containers to access the host's NVIDIA GPU.

## 3. Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Prepare your dataset:**
    Place your training data in a file named `dataset.jsonl` in the root directory. The file must be a JSON Lines file, where each line is a JSON object with `instruction`, `input`, and `output` keys.

3.  **Configure the Model (Optional):**
    You can specify the base model to use for fine-tuning.
    *   Create a `.env` file in the root directory (you can copy `.env.example`).
    *   Set the `MODEL_NAME` variable. For example:
        ```
        MODEL_NAME=unsloth/Qwen3-8B-128K-Instruct
        ```
    > **Note:** Only use models that are compatible with `unsloth` for fine-tuning. Do not use GGUF models for training.

## 4. Running the Application

### First-Time Run (Training)
The first time you run the application, it will automatically fine-tune the model using the `dataset.jsonl` file. This process can take a significant amount of time and will download the base model from the Hugging Face Hub.

```bash
docker-compose up
```

Once training is complete, the fine-tuned model will be saved to the `my-finetuned-model/` directory, and the application will run a default inference.

### Subsequent Runs (Inference)
On subsequent runs, the system will detect the existing model in `my-finetuned-model/` and will skip the training process, directly starting the inference service.

### Running Inference with a Custom Prompt
To ask your own questions, use the `docker-compose run` command:

```bash
docker-compose run --rm qlora-llm python3 src/inference.py "Your custom prompt here"
```
For example:
```bash
docker-compose run --rm qlora-llm python3 src/inference.py "Sancaktepe'de hangi parklar var?"
```

## 5. Customization

### Changing the System Prompt
The system prompt (the initial instruction that sets the model's persona and behavior) is hardcoded in the `src/inference.py` file. To change it, open the file and modify the text within the `alpaca_prompt` variable.

### Using a Different Dataset
To train on your own data, simply replace the `dataset.jsonl` file with your own data in the same JSON Lines format.

## 6. Exporting the Model to GGUF

After the model has been fine-tuned, you can export it to the highly optimized GGUF format, which is ideal for deployment with tools like Ollama or LM Studio.

Run the export script from your host machine (ensure you have the required packages from `requirements.txt` installed in a local python environment if you don't want to run it inside a container):
```bash
python src/export_gguf.py
```
This will create a `sancaktepe-model.gguf` file in your project's `output/` directory.
