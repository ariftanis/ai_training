In this project I want you to create a software code for creating docker containers for QLoRA LLM new data injection. Use knowledge below and create a better step by step TODO plan.

Given your powerful hardware (RTX 5090 with 32GB VRAM), you can efficiently use **QLoRA** (Quantized Low-Rank Adaptation):
- Loads the base model in 4-bit precision (low VRAM usage).
- Trains small "adapter" layers (LoRA) on your data.
- Achieves near full-fine-tuning quality with 70–90% less memory and fast training.

**Best tool: Unsloth** – It's optimized for NVIDIA GPUs (including RTX 50-series), 2–3x faster than standard Hugging Face methods, uses 70% less VRAM, and simplifies setup. It supports modern models like Llama 3.1 or Mistral.

**Recommended base model**: Meta-Llama-3.1-8B-Instruct (excellent for instruction-following and knowledge injection) or Mistral-7B-Instruct-v0.3 (fast and strong on domain knowledge). Both fit easily in your 32GB VRAM for QLoRA.

### Step-by-Step Guide to Fine-Tune Locally

#### Step 1: Set Up Your Environment
- Use a fresh Python environment (Python 3.10+ recommended).
- Install CUDA 12.x (matching your RTX 5090 drivers) from NVIDIA's site.
- Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl" "peft" "accelerate" "bitsandbytes"
pip install datasets transformers
```

- (Optional) For better speed: Install Flash Attention if supported.

#### Step 2: Prepare Your Dataset
Your .md files need to be converted into a supervised fine-tuning format (instruction-response pairs) for best results. This teaches the model to answer questions about your content reliably.

- Combine your .md files into one or split into chunks.
- Create a JSONL file (one JSON object per line) like this:

```json
{"messages": [{"role": "user", "content": "What is [specific question from your docs]?"}, {"role": "assistant", "content": "Accurate answer based on your markdown content..."}]}
```

- Aim for 500–5,000 high-quality examples (more is better, but quality matters most).
- Tools to help:
  - Use "alpaca" format (instruction/input/output) if pure knowledge injection. my documents are .md and .json files located in /documents folder I want you to use pure knowledge injection.

Upload as a Hugging Face dataset or load locally:

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="your_data.jsonl", split="train")
```

#### Step 3: Fine-Tune with Unsloth
Use this script (save as train.py and run with `python train.py`):

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Config
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Pre-quantized for speed
max_seq_length = 8192  # Adjust based on your content length
dataset = load_dataset("json", data_files="your_data.jsonl", split="train")

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect (bfloat16 on your GPU)
    load_in_4bit=True,  # QLoRA
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Rank (32–128; higher = better but more VRAM)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Or use packing for efficiency
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=4,  # Adjust (higher = faster, up to VRAM limit ~16 on 32GB)
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,  # Or num_train_epochs=3; monitor for overfitting
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",  # VRAM efficient
        report_to="none",
    ),
)

# Train!
trainer.train()
```

- Expected time: 1–10 hours depending on dataset size (your hardware is beast-mode).
- Monitor VRAM: Should stay under 20–25GB.

#### Step 4: Save and Merge the Model
```python
model.save_pretrained("my-finetuned-model")
tokenizer.save_pretrained("my-finetuned-model")

# Merge LoRA into base for faster inference (optional, uses more temp VRAM)
model.save_pretrained_merged("my-finetuned-merged", tokenizer, save_method="merged_16bit")
```

#### Step 5: Test and Use Your Model
- Run locally with Ollama:
  - Convert to GGUF (use llama.cpp tools).
  - Or use LM Studio/Text Generation WebUI.
- Inference example:

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(["Question about your docs?"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

- The model will now reliably recall and reason over your markdown content.

#### Tips for Best Results
- Avoid overfitting: Use a small validation set and stop early if loss plateaus.
- If pure knowledge injection (no Q&A): Use continued pre-training format (just raw text chunks).
- Larger model? Try Llama-3.1-70B with higher rank – your 32GB can handle QLoRA.
- Troubleshooting: Check Unsloth docs/GitHub for RTX 50-series notes.

This will give you a model that answers questions about your content accurately and consistently – far better than RAG. If issues arise, Unsloth's Colab notebooks are excellent for testing first. Enjoy your custom LLM!