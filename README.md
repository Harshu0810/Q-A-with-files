📄 Universal Drive RAG Chatbot (GPU-Accelerated)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from documents shared via public Google Drive links, using GPU-accelerated open-source LLMs on Google Colab.

This project demonstrates end-to-end RAG system design, including document ingestion, semantic indexing, caching, and an interactive chatbot UI.

🚀 Key Features

🔗 Universal Google Drive ingestion

Accepts public Drive file or folder links

Supports multiple files & nested folders

⚡ GPU-accelerated inference

Runs on Google Colab (free GPU, ~16GB RAM)

🧠 Retrieval-Augmented Generation (RAG)

Semantic embeddings + vector search + LLM synthesis

💬 Interactive Gradio chatbot UI

♻️ Index caching

Reuses indexes for repeated Drive links (faster queries)

🔐 No OpenAI / No API keys required

🧹 Ephemeral & privacy-safe

Documents are loaded temporarily and discarded after session end

🏗️ Architecture Overview

Google Drive Link (File / Folder)
        ↓

Temporary Runtime Ingestion
        ↓

Semantic Embeddings (Sentence-Transformers)
        ↓

Vector Store Index (LlamaIndex)
        ↓

GPU-based Open-Source LLM (Mistral-7B, 4-bit)
        ↓

Gradio Chatbot Interface

🧠 Technologies Used

LlamaIndex – RAG orchestration & indexing

Sentence-Transformers – Embeddings

Hugging Face Transformers – Open-source LLMs

BitsAndBytes – 4-bit quantization for GPU efficiency

Gradio – Chatbot UI

Google Colab – Free GPU runtime

📂 Example Use Cases

Research paper Q&A

Study notes summarization

Multi-PDF knowledge base chat

Technical documentation assistant

⚠️ Notes & Limitations

Google Drive links must be publicly accessible (viewer access)

Documents are temporarily downloaded into the Colab runtime

Free Colab sessions reset after inactivity
