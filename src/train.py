import logging
import sys
import psutil

# Set environment variables to avoid CUDA compilation issues
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is not required, just load environment variables from system
    pass

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import json
import sys
from typing import Tuple

# =================================================================================
# Configure logging to show download progress from huggingface_hub
# =================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# Enable detailed logging for transformers
import transformers
transformers.utils.logging.set_verbosity_info()

# Enable huggingface_hub progress bars
import huggingface_hub
huggingface_hub.utils.tqdm.disable = False
# =================================================================================


def clean_dataset(input_file: str, cleaned_file: str, rejected_file: str) -> Tuple[int, int]:
    """
    Cleans the dataset by separating valid and invalid lines.

    Args:
        input_file: Path to the input dataset file
        cleaned_file: Path to save the cleaned dataset
        rejected_file: Path to save the rejected lines

    Returns:
        Tuple of (valid_lines_count, rejected_lines_count)
    """
    print(f"Cleaning dataset: {input_file}")
    print(f"Valid entries will be saved to: {cleaned_file}")
    print(f"Rejected entries will be saved to: {rejected_file}")

    valid_count = 0
    rejected_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(cleaned_file, 'w', encoding='utf-8') as clean_outfile, \
         open(rejected_file, 'w', encoding='utf-8') as reject_outfile:

        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse the JSON
                data = json.loads(line)

                # Check if required fields exist and are strings
                required_fields = ['instruction', 'input', 'output']
                is_valid = True

                for field in required_fields:
                    if field not in data:
                        print(f"Line {line_num}: Missing required field '{field}'")
                        is_valid = False
                        break
                    elif not isinstance(data[field], str):
                        print(f"Line {line_num}: Field '{field}' is not a string (type: {type(data[field])})")
                        is_valid = False
                        break

                if is_valid:
                    # Write to cleaned dataset
                    clean_outfile.write(line + '\n')
                    valid_count += 1
                    if valid_count % 5000 == 0:
                        print(f"Processed {valid_count} valid entries...")
                else:
                    # Write to rejected dataset
                    reject_outfile.write(f"Line {line_num}: {line}\n")
                    rejected_count += 1

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                reject_outfile.write(f"Line {line_num}: {line}\n")
                rejected_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error - {e}")
                reject_outfile.write(f"Line {line_num}: {line}\n")
                rejected_count += 1

    return valid_count, rejected_count


# 1. Clean the dataset first
print(">>> Starting dataset cleaning process...")
valid_count, rejected_count = clean_dataset("dataset.jsonl", "cleaned_dataset.jsonl", "rejected_dataset.jsonl")

if rejected_count > 0:
    print(f"[WARNING] Found {rejected_count} invalid entries. Cleaned dataset saved to 'cleaned_dataset.jsonl'")
    print(f"Rejected entries saved to 'rejected_dataset.jsonl' for review and correction.")
else:
    print("[SUCCESS] No issues found. The dataset is clean and ready for training.")

# 2. Load the cleaned dataset
dataset = load_dataset("json", data_files="cleaned_dataset.jsonl", split="train")

# Add a validation step here
print("Dataset loaded successfully. Validating schema...")
# Check if all required columns are present and have the correct type
required_columns = {"instruction": "string", "input": "string", "output": "string"}
for col, expected_type in required_columns.items():
    if col not in dataset.column_names:
        raise ValueError(f"Missing required column: {col}")
    if str(dataset.features[col].dtype) != expected_type:
        raise ValueError(f"Column '{col}' has unexpected type: {dataset.features[col].dtype}, expected {expected_type}")

print("Dataset schema validated successfully.")

# 3. Model Configuration from environment variables
model_name = os.getenv("MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")  # Default to Llama 3.1 8B
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "8192"))  # Increased for larger model and context


# 4. Enhanced download progress tracking
print(">>> Preparing to download model and tokenizer...")
print(f">>> Model: {model_name}")

# Ensure progress bars are enabled (redundant call removed - already set in main config)
print(">>> Loading model and tokenizer...")
# The download progress should now be visible
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect (bfloat16 on your GPU)
    load_in_4bit=True,  # QLoRA
)
print(">>> Model and tokenizer loaded successfully.")

# Add EOS_TOKEN after tokenizer is loaded
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN at the end of each response

# Format the dataset for Alpaca format with EOS_TOKEN (do it once)
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN at the end, otherwise generation might go on forever!
        text = f"""### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""" + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
dataset = dataset.map(formatting_prompts_func, batched = True,)


# 5. Add LoRA adapters for optimal information injection from environment variables
lora_rank = int(os.getenv("LORA_RANK", "64"))  # Increased rank for better information injection with larger model
lora_alpha = int(os.getenv("LORA_ALPHA", "32"))  # Increased alpha for better learning
lora_dropout = float(os.getenv("LORA_DROPOUT", "0.1"))  # Small dropout to prevent overfitting

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
)

# Prepare for potential psutil usage in Unsloth compiled functions
dataset_num_proc = os.getenv("DATASET_NUM_PROC", "2")
os.environ["HF_DATASETS_NUM_PROC"] = dataset_num_proc  # Alternative way to set number of processes

# Make sure global namespace has psutil for Unsloth compiled files
import psutil
import builtins
builtins.psutil = psutil

# 6. Trainer for optimal information injection with environment variables
num_epochs = int(os.getenv("NUM_EPOCHS", "2"))  # Increased epochs for better information injection
batch_size = int(os.getenv("BATCH_SIZE", "1"))  # Adjusted for larger model to manage memory
gradient_accumulation = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))  # Increased for better gradient estimates
warmup_steps = int(os.getenv("WARMUP_STEPS", "50"))  # Increased for more stable training
learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))  # Standard learning rate for LoRA
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))  # Weight decay for regularization
output_dir = os.getenv("OUTPUT_DIR", "outputs")
save_strategy = os.getenv("SAVE_STRATEGY", "epoch")  # Save checkpoint at each epoch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Use the combined text field
    max_seq_length=max_seq_length,
    dataset_num_proc=int(dataset_num_proc),  # Add this parameter to bypass the psutil error docker my utilize 2 virtual codes generally
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,  # From environment variable
        gradient_accumulation_steps=gradient_accumulation,  # From environment variable
        warmup_steps=warmup_steps,  # From environment variable
        num_train_epochs=num_epochs,  # From environment variable
        learning_rate=learning_rate,  # From environment variable
        weight_decay=weight_decay,  # From environment variable
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,  # Balanced logging frequency
        output_dir=output_dir,  # From environment variable
        optim="adamw_8bit",  # Memory efficient optimizer
        report_to="none",
        remove_unused_columns=False,  # Important for custom datasets
        max_grad_norm=1.0,  # Gradient clipping for stability
        save_strategy=save_strategy,  # From environment variable
        save_steps=500,  # Additional save points
        gradient_checkpointing_kwargs={"use_reentrant": False},  # To prevent potential issues
    ),
)

# 7. Train!
print("\n>>> Starting model training...")
trainer.train()
print(">>> Training finished.")

# 8. Save the model
print("\n>>> Saving fine-tuned model...")
model.save_pretrained("my-finetuned-model")
tokenizer.save_pretrained("my-finetuned-model")
print(">>> Model saved to 'my-finetuned-model'")