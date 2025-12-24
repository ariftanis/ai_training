import torch
from unsloth import FastLanguageModel
import sys

def run_inference(prompt):
    """
    Loads the fine-tuned model and runs inference on the given prompt.
    """
    model_path = "my-finetuned-model"
    
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set the model to inference mode
    FastLanguageModel.for_inference(model)

    # Format the prompt using Alpaca format
    alpaca_prompt = f"""### Instruction:
You are a helpful assistant with knowledge about the Sancaktepe municipality. Based on the provided context, answer questions truthfully.

### Input:
{prompt}

### Response:
"""

    # Tokenize the prompt
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")

    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print only the response part
    response_start = response.find("### Response:") + len("### Response:")
    print(response[response_start:].strip())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        run_inference(prompt)
    else:
        print("Usage: python src/inference.py <your_prompt>")

