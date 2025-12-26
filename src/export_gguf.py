import os
import subprocess
import sys
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
    Merges the fine-tuned LoRA adapters and converts the model to GGUF format.
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

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    lora_model_path = "my-finetuned-model"
    merged_model_path = "my-finetuned-merged"
    gguf_output_name = os.getenv("GGUF_OUTPUT_NAME", "sancaktepe-model.gguf")
    gguf_output_path = os.path.join(output_dir, gguf_output_name)  # Save to output directory
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

    # Use Unsloth's built-in method for GGUF conversion if available (newer versions)
    try:
        model.save_pretrained_gguf(gguf_output_path, tokenizer, quantization_method="q8_0")
        print(f"✅ Successfully created GGUF model at: {gguf_output_path} using Unsloth's built-in method")
        print("You can now use this file with tools like Ollama or LM Studio.")
        return
    except Exception as e:
        # Check if it's specifically the AttributeError (method not available) or other errors
        if isinstance(e, AttributeError):
            print(">>> Unsloth's built-in GGUF conversion not available, using llama.cpp method...")
        else:
            print(f"Error during Unsloth's built-in GGUF conversion: {e}")
            print(">>> Falling back to manual llama.cpp method...")

        # Fallback to manual merging and conversion
        model.save_pretrained_merged(merged_model_path, tokenizer, save_method="merged_16bit")
        print(f"Model successfully merged and saved to '{merged_model_path}'")

        # --- 2. Set up llama.cpp for GGUF Conversion ---
        print(">>> Step 2: Setting up llama.cpp for GGUF conversion...")
        if not os.path.exists(llama_cpp_repo):
            print(f"'{llama_cpp_repo}' not found. Cloning from GitHub...")
            run_command("git clone https://github.com/ggerganov/llama.cpp.git")
        else:
            print(f"'{llama_cpp_repo}' directory already exists.")

        # Install llama.cpp dependencies
        print("Installing llama.cpp Python requirements...")
        run_command(f"pip install -r {os.path.join(llama_cpp_repo, 'requirements.txt')}")

        # --- 3. Convert to GGUF ---
        print(">>> Step 3: Converting the merged model to GGUF format...")
        convert_script_path = os.path.join(llama_cpp_repo, "convert.py")

        # We use f16 (float16) as it's a common and compatible type for GGUF conversion.
        # Other types like q4_k_m can be used for smaller file sizes.
        conversion_command = (
            f"python {convert_script_path} {merged_model_path}"
            f" --outfile {gguf_output_path}"
            f" --outtype f16" # Using f16 for high quality, can be changed to q4_k_m for smaller size
        )

        run_command(conversion_command)

        print(f"\n✅ Successfully created GGUF model at: {gguf_output_path}")
        print("You can now use this file with tools like Ollama or LM Studio.")


if __name__ == '__main__':
    export_to_gguf()
