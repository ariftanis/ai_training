Step-by-Step Plan to Improve Dataset Preparation for Better Factual RecallYour current Alpaca-style JSONL (one instruction per entry describing large sections) is too coarse for a ~500k token corpus, leading to poor memorization of specific entities like names, addresses, and lists. Fine-tuning works best when data is broken into many focused, repetitive question-answer pairs that directly query key facts. This forces the model to repeatedly "see" and output exact details during training.Follow this plan to create a higher-quality dataset emphasizing near-perfect recall of critical facts.Step 1: Extract and Clean Raw Text from Your FilesGather all .md and .json files into one directory.
Write a Python script to read them:For .md: Use markdown or strip headers/footage, keep structure (headings, lists, tables).
For .json: Load with json.load(), extract relevant text fields (e.g., "content" or "description").

Concatenate or process individually, but preserve sections (e.g., via headings).
Clean: Remove extra whitespace, fix encoding, handle tables (convert to text).
Tool: Use langchain loaders (e.g., DirectoryLoader with UnstructuredMarkdownLoader).

Goal: Produce clean text chunks or full documents for next steps.Step 2: Chunk the Documents SemanticallyAvoid large monolithic entries. Split into smaller, meaningful units (200–1000 tokens each).
Strategies:Semantic/propositional chunking: Split on paragraphs, list items, table rows, or headings. One chunk per fact/section (e.g., one for CEO bio, one for each depot address).
Add 20–30% overlap to preserve context.
Use libraries: LangChain's RecursiveCharacterTextSplitter (with Markdown separators) or SemanticChunker (with embeddings).

For lists (e.g., documents needed): Split into individual items.
Output: List of chunks, each a self-contained fact or small section.

Why? Smaller, focused chunks enable precise QA pairs for specific recall.Step 3: Identify and Prioritize Key Factual EntitiesManually or semi-automatically list critical facts needing perfect recall:Names (CEO, managers).
Addresses (service buildings, parks, official buildings, location names, department names, manager names, offices).
Lists (required documents, procedures).
Numbers, dates, etc.

Optional: Use an LLM (e.g., you can use our own local llm model my machine has ollama and hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M model you can use this model as an LLM engine for this kind of tasks ) to extract entities from chunks into a structured list (e.g., JSON of key-value pairs).

Goal: Ensure these facts get extra emphasis in data.Step 4: Generate Synthetic QA Pairs (Core Improvement)Use a strong LLM (e.g., you can use our own local llm model my machine has ollama and hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M model you can use this model as an LLM engine for this kind of tasks ) as "teacher" to create 5–20 diverse questions per chunk.
Prompt template example:

Context: {chunk_text}

Generate 10 diverse, realistic questions that can be answered directly from this context. Focus on specific facts like names, addresses, lists. Include variations (direct, paraphrased, scenario-based). For each question, provide the exact answer from the context only (no hallucination).
Output as JSON: [{"question": "...", "answer": "..."}, ...]

Emphasize key entities: Add instructions like "Generate at least 3 questions about names/addresses."
For lists: Questions like "What are all required documents for X?" or "List the supply depot addresses."
Generate thousands of pairs (aim for 5k–20k+ entries; synthetic data scales easily).
Tools: Unsloth's synthetic data notebook, LlamaIndex QA generator, or custom script with LangChain.

Why? Direct QA forces repeated exposure to exact facts → excellent memorization. Diversity prevents rote learning.Step 5: Augment for Repetition and RobustnessFor critical facts: Duplicate high-priority QA pairs 3–10 times (with slight paraphrasing).
Add negative/misconception pairs if needed (e.g., "Is the CEO named Wrong Name? → No, it's Correct Name.").
Mix formats: Some pure fact recall ("Who is the CEO?"), some reasoning ("Based on company info, where is the main depot?").

Step 6: Format as JSONL for TrainingUse ShareGPT or evolved Alpaca format (better for chat):json

{"messages": [{"role": "user", "content": "Question here"}, {"role": "assistant", "content": "Exact answer here"}]}

Or standard Alpaca: {"instruction": "Question", "input": "", "output": "Answer"}.
Include system prompt if needed (e.g., "You are an expert on company facts. Answer precisely.").

Step 7: Quality Control and DeduplicationFilter: Manually spot-check 100–200 pairs for accuracy (no hallucinations).
Auto-filter: Use LLM-as-judge to score faithfulness to context.
Deduplicate similar questions.
Split: 90% train, 10% validation (test recall on held-out facts).

Step 8: Train and EvaluateUse your QLoRA setup (rank 64–128, more epochs like 5–10).
Evaluate: Create test set of 100–200 unseen questions on key facts. Measure exact match accuracy.
If recall weak: Increase synthetic pairs for those facts, or add continued pretraining on raw chunks first.

This approach (chunking + massive synthetic QA) routinely achieves near-perfect factual recall on domain data, far better than broad descriptive entries. Start small (one document → 100 pairs) to test, then scale.If you share sample .md snippets or key facts, I can help craft exact prompts/scripts!

