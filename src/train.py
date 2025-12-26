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


# 1. Load the dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

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
num_epochs = int(os.getenv("NUM_EPOCHS", "3"))  # Increased epochs for better information injection
batch_size = int(os.getenv("BATCH_SIZE", "1"))  # Adjusted for larger model to manage memory
gradient_accumulation = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))  # Increased for better gradient estimates
warmup_steps = int(os.getenv("WARMUP_STEPS", "50"))  # Increased for more stable training
learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))  # Standard learning rate for LoRA
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