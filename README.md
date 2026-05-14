---
title: RAG Document Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 📄 RAG Document Chatbot

A local-first Retrieval-Augmented Generation chatbot that answers questions about your documents using **LlamaIndex**, **Streamlit**, and open-source LLMs.

## Features

- **Document Q&A** — Ask natural language questions and get grounded answers
- **Source Citations** — See exactly which pages informed each answer
- **Dual Deployment** — Runs locally (Ollama) or on HuggingFace Spaces (HF Inference API)
- **Fully Open Source** — No paid API keys required

## Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull a small Ollama model:**
   ```bash
   ollama pull qwen2.5:0.5b
   ```

3. **Place your PDFs** in the `data/` folder.

4. **Build the search index:**
   ```bash
   python build_index.py
   ```

5. **Run the chatbot:**
   ```bash
   streamlit run app.py
   ```

## Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with **Streamlit** SDK.
2. Push this repository to the Space.
3. Add your `HF_TOKEN` as a **Secret** in Space Settings.
4. The app will auto-detect the HF Spaces environment and use the HF Inference API.

## Configuration

All settings are configurable via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token (required for HF Spaces) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM model for HF Inference API |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Ollama model for local development |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `512` | Document chunk size for indexing |
