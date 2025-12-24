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
    
    # Step 1: Prepare the dataset
    echo "\n>>> Running dataset preparation..."
    python3 src/prepare_dataset.py
    
    # Step 2: Run the training
    echo "\n>>> Running model training..."
    python3 src/train.py
    
    echo "\n----------------------------------------------------"
    echo ">> Training complete. The model is now saved."
    echo ">> Future runs will skip training and run inference."
    echo "----------------------------------------------------"
else
    echo "----------------------------------------------------"
    echo ">> Trained model found. Skipping training."
    echo "----------------------------------------------------"
fi

# Execute the command passed to the container (e.g., run inference)
echo "\n>>> Executing container command..."
exec "$@"
