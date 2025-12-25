# Step-by-Step TODO Plan: QLoRA LLM Fine-Tuning and Dockerization

This plan outlines the process of preparing a dataset, fine-tuning a Large Language Model (LLM) using Unsloth's optimized approach, converting it to GGUF for efficient inference, and packaging it into a Docker container. For initial testing, the **Phi-3.5-mini-instruct model** will be used, specifically the `unsloth/Phi-3.5-mini-instruct` variant due to its small size for quick testing. Once validated, we can scale up to larger models like Phi-3-medium-4k-instruct.

## System Environment

*   **Operating System:** Ubuntu 24.04 LTS
*   **CUDA Support:** Fully enabled and supported by the system drivers. This allows for GPU acceleration during model training and inference.
*   **CUDA Base Image:** nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 with additional packages python3-dev and build-essential for compilation needs

### Phase 1: Project Setup and Data Preparation

1.  **Initialize Project Structure:**
    *   Ensure a `src` directory exists for Python scripts.
    *   Create necessary directories: `documents/` for source files, `outputs/` for model outputs.

2.  **Install Dependencies:**
    *   Install all necessary libraries using pip as per `requirements.txt`:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        pip install unsloth
        pip install psutil  # Added to resolve runtime issues with Unsloth compiled cache
        pip install --no-deps "trl" "peft" "accelerate" "bitsandbytes"
        pip install datasets transformers tqdm
        ```

3.  **Create Data Preparation Script (`src/prepare_dataset.py`):**
    *   This script reads all `.md` and `.json` files from the `documents/` directory.
    *   Converts them into the Alpaca format with EOS_TOKEN for proper training.
    *   Uses the format: "Instruction: {instruction}, Input: {input}, Response: {output}{EOS_TOKEN}"
    *   Saves the result to `dataset.jsonl` in the root directory.

### Phase 2: Model Training

4.  **Configure Model Parameters:**
    *   Set `max_seq_length = 2048` for initial tiny model testing (can be increased to 4096+ for larger models).
    *   Use `dtype = None` for auto-detection of bfloat16.
    *   Use `load_in_4bit = True` for memory-efficient QLoRA training.

5.  **Configure LoRA Parameters:**
    *   Set `r = 64` for LoRA rank (suggested values: 8, 16, 32, 64, 128) - can be adjusted for larger models.
    *   Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
    *   Use `lora_alpha = 16` and `lora_dropout = 0`.
    *   Use `use_gradient_checkpointing = "unsloth"` for VRAM optimization.

6.  **Create Training Script (`src/train.py`):**
    *   Load the `dataset.jsonl` with Alpaca formatting including EOS_TOKEN.
    *   Fine-tune the base model using QLoRA with proper dataset_num_proc setting to avoid psutil errors.
    *   Use SFTTrainer with appropriate parameters: batch size, gradient accumulation, learning rate.
    *   Save the resulting adapter weights to the `./my-finetuned-model` directory.

7.  **Run the Training Process:**
    *   Execute the training script: `python src/train.py` or via Docker: `docker-compose up`.
    *   Monitor training with logging_steps and progress tracking.
    *   Upon completion, verify that the `my-finetuned-model/` directory has been created with proper adapter files.

### Phase 3: Model Export and GGUF Conversion

8.  **Create GGUF Export Script (`src/export_gguf.py`):**
    *   This script will first merge the LoRA adapters from `my-finetuned-model/` into the base model using `save_pretrained_merged`.
    *   It will then automatically handle GGUF conversion using Unsloth's native GGUF support or llama.cpp.
    *   Creates a GGUF file (e.g., `sancaktepe-model.gguf`) in the project root.

9.  **Run the GGUF Export Process:**
    *   Execute the export script: `python src/export_gguf.py` after training is complete.
    *   Verify that the `.gguf` model file is created in the root directory.
    *   Supports various quantization methods: q4_k_m (recommended), q8_0, f16, etc.

### Phase 4: Dockerization and Deployment

10. **Update `requirements.txt`:**
    *   Ensure `psutil` is included in dependencies to resolve runtime issues.

11. **Update `Dockerfile`:**
    *   Use base image: `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`.
    *   Install `python3-dev` and `build-essential` for CUDA compilation requirements.
    *   Copy and install requirements before copying source code for better caching.
    *   Set up entrypoint script to handle first-time training vs inference mode.

12. **Create Inference Script (`src/inference.py`):**
    *   This script loads the fine-tuned model and runs inference in Alpaca format.
    *   Uses `FastLanguageModel.for_inference(model)` for 2x faster inference.
    *   Supports custom prompts via command-line arguments.

13. **Update `docker-compose.yml` file:**
    *   Configure GPU access via NVIDIA Container Toolkit.
    *   Set up volume mounts for persistent model storage and document access.
    *   Implement logic to skip training on subsequent runs when model exists.

14. **Update `entrypoint.sh`:**
    *   Check for existing model files before starting training.
    *   Run dataset preparation if training is needed.
    *   Execute training script if no model exists.
    *   Execute inference script as default command.

15. **Build and Test with Docker Compose:**
    *   Build and run the service using: `docker-compose up --build`.
    *   First run: prepares dataset, trains model, saves to persistent directory.
    *   Subsequent runs: skips training and runs inference directly.

### Phase 5: Documentation and Optimization

16. **Update Documentation:**
    *   Update `GEMINI.md` with current workflows and model choices.
    *   Document the Docker-based training and inference process.
    *   Explain how to run custom inference commands.

17. **Performance Optimization:**
    *   Use Unsloth's optimized settings for faster training and inference.
    *   Configure proper memory management with gradient checkpointing.
    *   Adjust batch sizes and accumulation steps based on available VRAM.
    *   Scale parameters appropriately when moving to larger models.

18. **Scalability:**
    *   Test with tiny model first to verify Docker setup works.
    *   Gradually increase model size and parameters as needed.
    *   Monitor resource usage and adjust accordingly.

19. **Troubleshooting:**
    *   Address psutil NameError by ensuring availability in all contexts.
    *   Handle CUDA compilation errors by including necessary development packages.
    *   Properly format datasets with EOS_TOKEN to avoid infinite generation issues.
