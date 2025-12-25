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

    # Format the prompt using the same Alpaca format used during training
    alpaca_prompt = f"""### Instruction:
You are a specialized AI assistant for Sancaktepe municipality services based on the Llama 3.1 8B model. Your role is to handle various types of user queries using a multi-agent approach. When users ask questions about municipal services, provide accurate information based on the context provided. For greetings, respond politely. For inappropriate queries, respond respectfully while redirecting to appropriate topics. Be helpful, accurate and concise in Turkish.
### Input:
{prompt}
### Response:
"""

    # Tokenize the prompt
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")

    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print only the response part
    response_start = response.find("### Response:") + len("### Response:")
    response_content = response[response_start:].strip()
    print(response_content)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        run_inference(prompt)
    else:
        print("Usage: python src/inference.py <your_prompt>")

