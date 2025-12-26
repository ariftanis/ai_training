# Data Preparation Improvement Plan - TODO List

This plan outlines steps to improve the dataset preparation process to achieve better factual recall, especially for important information like names, addresses, lists, etc. The current Alpaca-style JSONL format is too coarse for effective memorization of specific entities.

## Phase 1: Data Extraction and Cleaning

**Tasks:**
1. **[TODO]** Write Python script to extract and clean raw text from all .md and .json files in the documents/ directory
   - For .md files: Use markdown libraries to preserve structure (headings, lists, tables) while stripping unnecessary formatting
   - For .json files: Extract relevant text fields (e.g., "content", "description", "text")
   - Clean text: Remove extra whitespace, fix encoding issues, convert tables to readable text format

2. **[TODO]** Implement DirectoryLoader with UnstructuredMarkdownLoader to gather all files
   - Process .md and .json files into a unified format
   - Preserve document sections and headings as context

3. **[TODO]** Create a validation script to ensure all files are properly read and cleaned
   - Check for encoding issues
   - Verify text extraction completeness

## Phase 2: Semantic Chunking

**Tasks:**
4. **[TODO]** Implement semantic chunking using RecursiveCharacterTextSplitter
   - Target chunk sizes: 200-1000 tokens per chunk
   - Use markdown-aware separators to respect document structure
   - Add 20-30% overlap between chunks to preserve context

5. **[TODO]** Create chunking strategy for different content types
   - One chunk per fact/section (e.g., one for department head bio, one for each office address)
   - Split lists into individual, focused chunks
   - Handle tables by converting to structured text within chunks

6. **[TODO]** Validate chunk quality
   - Ensure each chunk is self-contained with sufficient context
   - Verify chunks maintain semantic coherence

## Phase 3: Critical Facts Identification

**Tasks:**
7. **[TODO]** Manually identify critical factual entities requiring perfect recall
   - Names (mayor, department heads, managers, officials)
   - Addresses (offices, service locations, buildings, parks)
   - Lists (required documents, procedures, contact numbers)
   - Important dates, numbers, and contact information

8. **[TODO]** Create structured list of key entities
   - Organize into categories (names, addresses, lists, etc.)
   - Prioritize by importance for municipal services

9. **[TODO]** (Optional) Use LLM to assist with entity extraction
   - Use available local model (Qwen3-30B-A3B-Instruct) as an LLM engine
   - Extract entities into structured JSON format

## Phase 4: Synthetic QA Generation

**Tasks:**
10. **[TODO]** Develop prompt template for synthetic QA generation
    - Template: Context: {chunk_text}\nGenerate 10 diverse, realistic questions that can be answered directly from this context...
    - Focus on specific facts like names, addresses, lists
    - Include variations (direct, paraphrased, scenario-based questions)

11. **[TODO]** Generate 5-20 diverse questions per chunk using LLM
    - Target: 5k-20k+ QA pairs total
    - Emphasize key entities with special instructions
    - Include: "Generate at least 3 questions about names/addresses" type instructions

12. **[TODO]** Create QA pairs for different question types
    - Direct recall: "Who is the mayor of Sancaktepe?"
    - List questions: "What are all required documents for residence permit?"
    - Scenario-based: "If someone wants to report a street light issue, which department should they contact?"

## Phase 5: Data Augmentation and Repetition

**Tasks:**
13. **[TODO]** Duplicate high-priority QA pairs for critical facts (3-10 times with slight variations)
   - Apply paraphrasing to avoid exact duplicates
   - Maintain factual accuracy

14. **[TODO]** Add negative/misconception handling pairs
   - Example: "Is the mayor named Wrong Name? → No, it's Correct Name"
   - Helps model learn accurate information by contrast

15. **[TODO]** Create different question formats
   - Pure fact recall: "Who is the head of parks department?"
   - Reasoning format: "Based on municipal structure, which office handles park maintenance?"

## Phase 6: Format Conversion and Quality Control

**Tasks:**
16. **[TODO]** Convert data to proper JSONL format for training
    - Use ShareGPT format: {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
    - Include system prompt: "You are an expert on Sancaktepe municipality facts. Answer precisely."

17. **[TODO]** Implement quality control measures
    - Manual spot-check of 100-200 QA pairs for accuracy
    - Auto-filter using LLM-as-judge to score faithfulness to context
    - Remove hallucinated answers

18. **[TODO]** Implement deduplication
    - Remove similar or duplicate questions
    - Keep only high-quality, diverse QA pairs

19. **[TODO]** Create train/validation split
    - 90% training data
    - 10% validation (hold-out test set for fact recall evaluation)

## Phase 7: Implementation and Testing

**Tasks:**
20. **[TODO]** Create new dataset preparation script (improved prepare_dataset.py)
    - Implement all the above steps in a single pipeline
    - Add logging and progress indicators
    - Include error handling and data validation

21. **[TODO]** Test with small dataset first
    - Process a single document into ~100 QA pairs
    - Train small model to validate approach
    - Evaluate factual recall on this small test

22. **[TODO]** Scale up to full dataset
    - Process all documents following validated approach
    - Generate full QA dataset (5k-20k+ pairs)
    - Validate output quality before full training

## Phase 8: Evaluation and Iteration

**Tasks:**
23. **[TODO]** Create evaluation test set of 100-200 unseen questions on key facts
    - Focus on names, addresses, lists that need perfect recall
    - Measure exact match accuracy

24. **[TODO]** Train model with improved dataset and evaluate
    - Use current QLoRA setup with increased epochs (5-10)
    - Compare factual recall against previous model
    - Identify any remaining weak areas

25. **[TODO]** Iterate and improve based on evaluation results
    - Generate additional QA pairs for weak areas
    - Consider continued pretraining on raw chunks if needed
    - Optimize hyperparameters for the new dataset format