You can use this information for using LLM for generating relevant questions for data prep chunks. 
Path: src/validate_dataset.pyIn local machine there is installed ollama service with model loaded you can integrate 
this feature to create a better working prepare_dataset.py


this is the python filecode for you can use ollama llm prompts
---LLM_PROMPT code begin---
def generate_response_with_guidance(prompt: str) -> str:
    # Use .env first, fallback to Docker host (works on Windows/Mac/Windows)
    base_url = os.getenv('OLLAMA_API_BASE_URL', 'http://host.docker.internal:11434').rstrip('/')
    url = f"{base_url}/api/chat"  # Ollama endpoint (also works with LM Studio in Ollama compatibility mode)

    OLLAMA_MODEL_NAME = os.getenv(
        'OLLAMA_MODEL_NAME',
        'hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M'
    )

    # === Recommended parameters for this exact model (non-thinking) ===
    options = {
        "temperature": 0.6,             # 0.7 default
        "top_p": 0.8,                   # 0.8 default
        "top_k": 20,                    # 20 default
        "min_p": 0.01,                  # 0.00 default
        "presence_penalty": 0.1,        # light repetition control, very safe between 0-2
        "num_ctx": 16384,               # output context token length full context window
        "num_predict": -1,              # no hard token limit (will respect 16384)
        "repeat_penalty": 1.05,         # additional safety net (optional but helpful)
    }

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": options,
        # Explicitly disable thinking mode if your Ollama/LM Studio supports it
        "format": "",                   # some servers expect this
        "keep_alive": -1                # keep model loaded
    }

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'curl/7.81.0',   # mimics curl – helps with some servers
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        data = response.json()

        # Unified response parsing (works with Ollama and OpenAI-compatible servers)
        if "choices" in data and len(data["choices"]) > 0:
            # OpenAI / LM Studio format
            markdown_text = data["choices"][0]["message"]["content"]
        elif "message" in data and "content" in data["message"]:
            # Pure Ollama format
            markdown_text = data["message"]["content"]
        else:
            markdown_text = str(data)

        # Remove any leftover <think>...</think> blocks (just in case)
        clean_text = re.sub(r'<think>[\s\S]*?</think>', '', markdown_text, flags=re.IGNORECASE).strip()

        # Convert to safe HTML
        html = markdown.markdown(clean_text, extensions=['nl2br', 'fenced_code', 'tables'])
        sanitized_html = bleach.clean(
            html,
            tags=['p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'strong', 'em', 'b', 'i', 'ul', 'ol', 'li',
                  'a', 'code', 'pre', 'blockquote', 'hr', 'img'],
            attributes={'a': ['href', 'title'], 'img': ['src', 'alt', 'title']},
            strip=False
        )

        return sanitized_html

    except requests.exceptions.RequestException as e:
        return f"<p style='color:red;'>Ollama/LM Studio API error: {e}</p>"
    except Exception as e:
        return f"<p style='color:red;'>Unexpected error: {e}</p>"
---LLM_PROMPT code end---