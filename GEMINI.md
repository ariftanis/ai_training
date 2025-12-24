# Project: Fine-tuning a Turkish Municipality LLM

This project aims to fine-tune a Large Language Model (LLM) to create a specialized chatbot or information retrieval system about the Sancaktepe municipality in Istanbul, Turkey.

## Project Overview

The primary goal is to leverage the documents in the `documents/` directory to train a language model that can answer questions and provide information about the municipality's services, regulations, and general information.

The `plan.md` file outlines a detailed technical plan for this process, which involves:

*   **Model:** Using a base model like `Meta-Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct-v0.3`.
*   **Fine-tuning method:** Employing QLoRA (Quantized Low-Rank Adaptation) for efficient training.
*   **Tooling:** Utilizing the `Unsloth` library for optimized performance on NVIDIA GPUs.

## Dataset

The `documents/` directory contains a collection of Markdown (`.md`) files that serve as the knowledge base for the LLM. These files cover a wide range of topics related to the Sancaktepe municipality, including:

*   **General Information:** Mission, vision, and general details about the municipality.
*   **Services:** Descriptions of services provided by the municipality, such as health, social support, and cultural activities.
*   **Regulations (`Yönetmelikler`):** Official regulations governing the municipality's directorates.
*   **Directorates (`Müdürlükler`):** Information about the various directorates and their functions.
*   **Projects:** Details on completed and ongoing projects.
*   **Facilities:** Information about parks, health centers, and other municipal facilities.

## Training Plan

The `plan.md` file provides a step-by-step guide for the fine-tuning process:

1.  **Environment Setup:** Install Python, CUDA, and the necessary Python libraries (`torch`, `unsloth`, `transformers`, etc.).
2.  **Data Preparation:** Convert the Markdown files into a supervised fine-tuning format (e.g., JSONL with instruction-response pairs).
3.  **Fine-tuning:** Run the training script (`train.py` in `plan.md`) using `Unsloth` and `SFTTrainer`.
4.  **Model Saving and Merging:** Save the fine-tuned model and merge the LoRA adapters for faster inference.
5.  **Testing and Inference:** Test the model's performance and use it for inference.

## Usage

The contents of this directory are intended to be used as a dataset for training a specialized LLM. The `plan.md` file serves as a technical guide for this process. The ultimate goal is to create a model that can act as a knowledgeable assistant for information related to the Sancaktepe municipality.
