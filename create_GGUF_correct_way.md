I have finetuned a model output results seems like a folder named sancaktepe-model.gguf when I double click and look inside there are .cache folder chat_template.jinja config.json model.safetensors.index.json model-00001-of00004.safetensors model-00002-of00004.safetensors model-00003-of00004.safetensors model-00004-of00004.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json file I seedid it correct? I tried to load this model with LM studio and it failed to index how can I create a correct gguf model file?

Searching the web

44 results

No, that's not a correct GGUF file – what you have in the "sancaktepe-model.gguf" folder is a standard Hugging Face (HF) model directory in safetensors format. This is the typical output from fine-tuning tools like PEFT/LoRA, TRL/SFTTrainer, or Axolotl. It contains:Split weight files (model-00001-of-00004.safetensors etc.)
Index file (model.safetensors.index.json)
Config, tokenizer files, etc.

GGUF is a completely different binary format (a single file like my-model-Q4_K_M.gguf) optimized for local inference with tools like llama.cpp (which LM Studio uses under the hood). LM Studio primarily supports GGUF files for local loading – it doesn't directly load raw HF safetensors folders (that's why it "failed to index").How to Create a Proper GGUF File from Your Fine-Tuned ModelImportant: If you fine-tuned using LoRA/PEFT adapters (common for efficiency), you must first merge the adapters into the base model to get full weights. Converting adapters directly won't work well.Step 1: Merge (if using LoRA/PEFT)Use this script (save as merge.py):python

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_dir = "path/to/your/sancaktepe-model.gguf"  # Your folder
output_dir = "path/to/merged-model"

model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype="auto")
model = model.merge_and_unload()

model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.save_pretrained(output_dir)

Run: python merge.pyThis creates a new folder (merged-model) with full (non-adapter) safetensors weights.If you did full fine-tuning (no LoRA), your folder already has full weights – skip merge and use it directly.Step 2: Convert to GGUF Using llama.cppClone the latest llama.cpp repo:

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

Install requirements:

pip install -r requirements/requirements-convert_hf_to_gguf.txt

(Optional but recommended) Build the tools:

make  # Or use CMake for full build if you want quantization binaries

Convert to unquantized GGUF (f16 is common – good quality, smaller than f32):

python convert_hf_to_gguf.py /path/to/merged-model --outfile my-finetuned-model-f16.gguf --outtype f16

Replace /path/to/merged-model with your merged (or original) folder.
This produces a single my-finetuned-model-f16.gguf file.

Quantize for smaller size/better performance (recommended for local use):
After building llama.cpp (make), run:

./llama-quantize my-finetuned-model-f16.gguf my-finetuned-model-Q4_K_M.gguf Q4_K_M

Popular options: Q4_K_M (great balance), Q5_K_M (higher quality), Q8_0 (near-original).
Delete the f16 version afterward to save space.

Step 3: Load in LM StudioUse the lms import CLI method (easiest):

lms import /path/to/my-finetuned-model-Q4_K_M.gguf

(It will prompt for publisher/name, or use --user-repo yourname/my-model.)
Or manually place the .gguf file in LM Studio's models folder with subfolders: models/yourname/my-finetuned-model/my-finetuned-model-Q4_K_M.gguf
Restart/refresh LM Studio – it should appear and load.

Tips & Common IssuesEnsure your base model is supported by llama.cpp (e.g., Llama, Mistral, Phi, Gemma, Qwen – most popular ones are).
If conversion fails (e.g., "architecture not supported"), check the model's config.json for the architecture.
For large models, conversion needs RAM (roughly model size + overhead).
Test prompts that match your fine-tune data to verify it worked.

This process works reliably for most fine-tunes. If you hit errors, share the exact error message/base model name for more help!

