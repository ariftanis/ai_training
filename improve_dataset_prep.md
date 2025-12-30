### Step-by-Step Guide to Preparing a High-Quality QLoRA Training Dataset from PDF and Markdown Documents

QLoRA (Quantized Low-Rank Adaptation) is an efficient fine-tuning method that excels with **high-quality, small-to-medium-sized datasets** (typically 1,000–50,000 examples). The key to making your fine-tuned LLM accurately answer domain-specific questions is creating an **instruction-tuning dataset** focused on question-answering (QA) over your knowledge sources. This teaches the model to reason about and retrieve information from your domain.

Industry standards (from sources like the original QLoRA paper, Alpaca, and modern practices) emphasize:
- **Quality over quantity**: Clean, diverse, factual QA pairs outperform large noisy datasets.
- Instruction format: Use structured prompts (e.g., Alpaca-style) to teach the model to follow questions and provide accurate responses.
- Domain fidelity: Answers must be grounded in your documents to avoid hallucinations.

Follow these steps to transform your PDFs and .md files into a ready-to-use dataset.

#### Step 1: Extract Clean Text from Your Documents
Raw PDFs and Markdown files often contain noise (headers, footers, tables, images, formatting artifacts).

**Tools and Methods**:
- For **Markdown (.md)**: Simple—read files directly with Python:
  ```python
  with open('file.md', 'r', encoding='utf-8') as f:
      text = f.read()
  ```
- For **PDFs**:
  - Use **Unstructured.io** (open-source, excellent for preserving structure):
    ```bash
    pip install unstructured
    ```
    ```python
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf("file.pdf", strategy="hi_res")  # Handles tables/images better
    text = "\n\n".join([str(el) for el in elements])
    ```
  - Alternatives: PyMuPDF (`fitz`), pdfplumber (good for tables), or pymupdf4llm (converts to Markdown).
- Batch process all files into a list of clean text chunks.

**Best Practices**:
- Remove boilerplate (e.g., page numbers, copyrights).
- Preserve structure: Keep sections, headings, and lists.
- Output: A corpus of plain text strings (one per document or section).

#### Step 2: Chunk the Text into Manageable Pieces
LLMs have context limits (e.g., 4k–8k tokens). Chunking ensures each piece fits while retaining meaning.

**Guidelines**:
- Chunk size: 512–2048 tokens (aim for 1000–2000 characters per chunk).
- Overlap: 20–50% overlap between chunks for context continuity.
- Semantic chunking: Split on paragraphs, sections, or headings (better than fixed-size).

**Tools**:
- LangChain or LlamaIndex for semantic chunking:
  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  chunks = splitter.split_text(full_text)
  ```
- Save chunks as a list or JSON for the next step.

**Output**: 100s–1000s of coherent text chunks representing your knowledge base.

#### Step 3: Generate Synthetic QA Pairs (The Core of High-Quality Data)
Manually creating QA is slow and expensive. Use a strong LLM (e.g., GPT-4o, Claude 3.5, or Llama 3.1 70B) to generate diverse, factual questions and answers grounded in each chunk.

**Why Synthetic?**
- Proven effective (e.g., Stanford Alpaca used self-instruct; QLoRA paper stresses small high-quality sets).
- Generates varied question types: factual, reasoning, multi-hop.

**Prompt Engineering for Generation** (Critical for Quality):
Use a few-shot prompt like this (adapt for your domain):

```
You are an expert in [your domain, e.g., finance/regulations]. Generate 5-10 diverse question-answer pairs based ONLY on the following text. Questions should be realistic user queries. Answers must be direct, accurate, and quoted/paraphrased from the text only—no hallucinations.

Text: {chunk}

Output ONLY JSON:
[
  {"question": "What is X?", "answer": "Y as per the text."},
  ...
]
```

- Vary question styles: "What/How/Why/Explain/Summarize/List".
- Instruct: "Base answers strictly on the text", "Be concise yet complete", "Include citations if possible".

**Tools for Automation**:
- **Open-source**:
  - Bonito (Mistral-based, efficient for document-to-instruction).
  - Meta's Synthetic Data Kit (handles PDFs directly → QA pairs).
  - Unsloth's Synthetic Data notebook (parses PDFs, generates QA with Llama).
- **Commercial/Easy**:
  - OpenAI API (GPT-4o) or Anthropic (Claude) batch processing.
  - H2O LLM DataStudio or Argilla for curation.
- Generate 5–20 pairs per chunk → Aim for 1,000–10,000 total pairs.

**Best Practices**:
- Diversity: Mix simple/complex questions.
- Avoid bias/toxicity.
- Grounding: Enforce "only from text" to prevent hallucinations.

#### Step 4: Clean, Filter, and Curate the Dataset
Raw synthetic data needs refinement.

**Actions**:
- **Deduplicate**: Remove identical/redundant pairs.
- **Filter Quality**:
  - Use an LLM judge to score for accuracy, relevance, fluency (keep >8/10).
  - Manual review: Sample 10–20% for errors.
- **Balance**: Ensure coverage across documents/topics.
- **Augment**: Add variations (paraphrase questions) or negative examples if needed.

**Tools**: Argilla (UI for review), pandas for scripting.

#### Step 5: Format the Dataset in Instruction-Tuning Style
Industry standard for QLoRA/instruction fine-tuning is **Alpaca format** (JSON list of dicts).

**Recommended Structure** (Alpaca-style):
```json
[
  {
    "instruction": "Answer the question based on the provided context.",
    "input": "Context: {chunk}\n\nQuestion: {question}",
    "output": "{answer}"
  },
  ...
]
```
- Or simpler chat format (for models like Llama-3):
```json
[
  {
    "messages": [
      {"role": "user", "content": "{question}\n\nContext: {chunk}"},
      {"role": "assistant", "content": "{answer}"}
    ]
  },
  ...
]
```

**Alternatives**:
- Include context in input (as above) for better grounding.
- Pure QA without context if chunks are short (but including context improves domain accuracy).

**Final File**: Save as `dataset.json` or `dataset.jsonl` (one JSON per line).

**Size Recommendation**: Start with 1,000–5,000 high-quality examples. More isn't always better—focus on quality.

#### Step 6: Validate and Test
- Split: 90% train, 10% validation.
- Manual check: Test 50 examples—ensure answers are factual.
- Optional: Evaluate generated pairs with metrics (e.g., ROUGE/BERTScore against ground truth).

#### Bonus Tips for Success
- **Quantity vs. Quality**: QLoRA paper shows small high-quality datasets match larger ones.
- **Avoid Common Pitfalls**: No hallucinations in answers, diverse questions, consistent formatting.
- **Next Step (Fine-Tuning)**: Use libraries like PEFT + TRL (SFTTrainer) or Unsloth for efficient QLoRA training.
- If stuck on generation: Start with OpenAI API for quick high-quality synth data.

This process will yield a dataset that teaches your fine-tuned LLM to reliably answer domain questions. Expect strong results even on consumer GPUs! If your domain is sensitive, use local/open models for generation.
