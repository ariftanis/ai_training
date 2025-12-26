import os
import subprocess
import sys
import torch
from unsloth import FastLanguageModel

def run_command(command):
    """Executes a shell command and prints its output in real-time."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, encoding='utf-8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc != 0:
        print(f"Error: Command '{command}' failed with exit code {rc}")
        sys.exit(1)
    return rc

def export_to_gguf():
    """
    Merges the fine-tuned LoRA adapters and converts the model to proper GGUF format.
    """
    # Set environment variable to accept system package installations automatically
    import os
    os.environ["UNSLOTH_INSTALL_FORCE_ACCEPT"] = "1"

    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # dotenv is not required, just load environment variables from system
        pass

    # Get quantization method from environment variable
    quantization_method = os.getenv("GGUF_QUANTIZATION", "q8_0")

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    lora_model_path = "my-finetuned-model"
    merged_model_path = "my-finetuned-merged"
    gguf_output_name = os.getenv("GGUF_OUTPUT_NAME", "sancaktepe-model.gguf")
    llama_cpp_repo = "llama.cpp"

    # --- 1. Merge LoRA Adapters ---
    print(">>> Step 1: Merging LoRA adapters...")
    if not os.path.exists(lora_model_path):
        print(f"Error: The fine-tuned model directory '{lora_model_path}' does not exist.")
        print("Please run the training script (src/train.py) first.")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_model_path,
        dtype=None,
        load_in_4bit=True,
    )

    # Use Unsloth's built-in method for GGUF conversion with proper quantization
    try:
        print(f">>> Using quantization method: {quantization_method}")
        gguf_output_path = os.path.join(output_dir, gguf_output_name)
        model.save_pretrained_gguf(gguf_output_path, tokenizer, quantization_method=quantization_method)
        print(f"✅ Successfully created GGUF model at: {gguf_output_path} using Unsloth's built-in method with {quantization_method} quantization")
        print("You can now use this file with tools like Ollama or LM Studio.")
        return
    except Exception as e:
        print(f"Error during Unsloth's built-in GGUF conversion: {e}")
        print(">>> Falling back to manual method based on your working approach...")

        # --- Manual method following your successful approach ---
        print(">>> Step 2: Loading model from adapter using standard transformers...")
        from transformers import AutoModelForCausalLM

        # Merge the LoRA adapters into the base model
        model.save_pretrained_merged(merged_model_path, tokenizer, save_method="merged_16bit")
        print(f">>> Model merged successfully to: {merged_model_path}")

        # Now use the successful approach: load with transformers and save merged
        print(">>> Step 3: Loading merged model with transformers...")
        merged_model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Save in standard format
        final_merged_path = os.path.join(output_dir, "merged-model")
        print(f">>> Saving merged model to: {final_merged_path}")
        merged_model.save_pretrained(final_merged_path)
        tokenizer.save_pretrained(final_merged_path)

        # --- Step 4: Convert to GGUF using llama.cpp ---
        print(">>> Step 4: Setting up llama.cpp for conversion...")
        if not os.path.exists(llama_cpp_repo):
            print(f"'{llama_cpp_repo}' not found. Cloning from GitHub...")
            run_command("git clone https://github.com/ggerganov/llama.cpp.git")
        else:
            print(f"'{llama_cpp_repo}' directory already exists.")

        # Install llama.cpp dependencies
        print("Installing llama.cpp Python requirements...")
        run_command(f"pip install -r {os.path.join(llama_cpp_repo, 'requirements.txt')}")

        # Run the conversion command exactly as you did successfully
        gguf_output_path = os.path.join(output_dir, f"sancaktepe-model-{quantization_method.upper()}.gguf")
        print(f">>> Converting to GGUF format: {gguf_output_path}")

        convert_script_path = os.path.join(llama_cpp_repo, "convert_hf_to_gguf.py")
        conversion_cmd = f"python {convert_script_path} {final_merged_path} --outfile {gguf_output_path} --outtype {quantization_method.lower()}"

        run_command(conversion_cmd)

        print(f"\n✅ Successfully created quantized GGUF model at: {gguf_output_path}")
        print("You can now use this file with tools like Ollama or LM Studio.")


if __name__ == '__main__':
    export_to_gguf()
