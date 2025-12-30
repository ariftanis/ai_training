from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "sancaktepe-model.gguf")  # This is the folder
output_dir = os.path.join(current_dir, "merged-model")  # Output folder

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

try:
    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",       # Automatically assign layers to GPU/CPU
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        low_cpu_mem_usage=True   # Reduce RAM usage during loading
    )

    print("💾 Saving model to:", output_dir)
    model.save_pretrained(output_dir)

    print("🔠 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)

    print("🎉 Model and tokenizer saved successfully!")

except Exception as e:
    print(f"❌ Error occurred: {e}")