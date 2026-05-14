"""
Centralized configuration for the RAG Chatbot.
Auto-detects deployment environment (local vs HuggingFace Spaces).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Deployment Detection ─────────────────────────────────────────────────────
# HuggingFace Spaces automatically sets the SPACE_ID env var
IS_HF_SPACE = os.getenv("SPACE_ID") is not None

# ── Paths ─────────────────────────────────────────────────────────────────────
PERSIST_DIR = "storage"
DATA_DIR = "data"
UPLOAD_DIR = "uploads"

# ── Model Configuration ──────────────────────────────────────────────────────
# Embedding model (runs locally on both environments — ~80MB, very fast)
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LLM for HuggingFace Spaces (free serverless Inference API)
HF_MODEL_NAME = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-72B-Instruct"
)
HF_TOKEN = os.getenv("HF_TOKEN")

# LLM for local development via Ollama (smallest model for low-resource systems)
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Chunking (single source of truth) ────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Query Engine Settings ─────────────────────────────────────────────────────
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120.0"))
