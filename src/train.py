import logging
import sys
import psutil
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

# 2. Format the dataset for Alpaca format
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"""### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        texts.append(text)
    return { "text" : texts, }
pass
dataset = dataset.map(formatting_prompts_func, batched = True,)


# 3. Model Configuration
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Pre-quantized for speed
max_seq_length = 8192  # Adjust based on your content length


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

# 5. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Rank (32–128; higher = better but more VRAM)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
)

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Use the combined text field
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # Add this parameter to bypass the psutil error ryzen 9800x has 8 cores
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        report_to="none",
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