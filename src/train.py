import logging
import sys
import psutil

# Set environment variables to avoid CUDA compilation issues
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

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

# 3. Model Configuration
model_name = "unsloth/Phi-3.5-mini-instruct"  # Tiny model for quick testing
max_seq_length = 2048  # Reduced for smaller model


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


# 5. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Reduced rank for the tiny model to save memory
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
)

# Prepare for potential psutil usage in Unsloth compiled functions
import os
os.environ["HF_DATASETS_NUM_PROC"] = "2"  # Alternative way to set number of processes

# Make sure global namespace has psutil for Unsloth compiled files
import psutil
import builtins
builtins.psutil = psutil

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Use the combined text field
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # Add this parameter to bypass the psutil error docker my utilize 2 virtual codes generally
    args=TrainingArguments(
        per_device_train_batch_size=1,  # Reduced batch size to save memory for tiny model
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        warmup_steps=5,  # Reduced for faster testing
        num_train_epochs=1,  # Reduced to 1 epoch for testing
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,  # More frequent logging for faster feedback
        output_dir="outputs",
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,  # Important for custom datasets
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