#!/bin/bash
set -e

MODEL_FILE="/app/my-finetuned-model/adapter_model.safetensors"

# Check if the fine-tuned model exists.
if [ ! -f "$MODEL_FILE" ]; then
    echo "----------------------------------------------------"
    echo ">> Model not found at $MODEL_FILE."
    echo ">> Starting the one-time training process now."
    echo ">> This will take a significant amount of time."
    echo "----------------------------------------------------"

    # Clean the model directory to ensure a fresh start
    echo
    echo ">>> Cleaning model directory for fresh start..."
    rm -rf /app/my-finetuned-model/*

    # Step 1: Prepare the dataset
    echo
    echo ">>> Running dataset preparation..."
    python3 src/prepare_dataset.py

    # Step 2: Run the training
    echo
    echo ">>> Running model training..."
    python3 src/train.py

    echo
    echo "----------------------------------------------------"
    echo ">> Training complete. The model is now saved."
    echo ">> Future runs will skip training and run inference."
    echo "----------------------------------------------------"

    # Step 3: Export the model to GGUF format for broader compatibility
    echo
    echo ">>> Converting model to GGUF format for broader compatibility..."
    python3 src/export_gguf.py
    echo
    echo "----------------------------------------------------"
    echo ">> GGUF export complete. Model is available as sancaktepe-model.gguf"
    echo "----------------------------------------------------"
else
    echo "----------------------------------------------------"
    echo ">> Trained model found. Skipping training."
    echo "----------------------------------------------------"
fi

# Execute the command passed to the container (e.g., run inference)
echo
echo ">>> Executing container command..."
exec "$@"
