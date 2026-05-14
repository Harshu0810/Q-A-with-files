"""
RAG Engine — Loads the vector index and creates a query engine.
Supports dual-mode LLM: Ollama (local) or HuggingFace Inference API (HF Spaces).

Exposes composable building blocks:
  - get_embed_model()       → embedding model
  - get_llm()               → LLM (auto-selects based on environment)
  - build_index_from_dir()  → build + persist index from a directory of documents
  - load_existing_index()   → load a previously persisted index
  - create_query_engine()   → wrap an index into a query engine
  - load_rag_engine()       → convenience: load existing index + create query engine
"""

import os
import shutil
from config import (
    IS_HF_SPACE,
    PERSIST_DIR,
    DATA_DIR,
    EMBED_MODEL_NAME,
    HF_MODEL_NAME,
    HF_TOKEN,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    REQUEST_TIMEOUT,
)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ── Building Blocks ──────────────────────────────────────────────────────────


def get_embed_model():
    """Load the local HuggingFace embedding model."""
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def get_llm():
    """
    Return the appropriate LLM based on the deployment environment.
    - HF Spaces  → HuggingFace Inference API (free, serverless)
    - Local      → Ollama (requires ollama to be running)
    """
    if IS_HF_SPACE:
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is required for HuggingFace Spaces deployment. "
                "Add it as a Secret in your Space's Settings → Variables and secrets."
            )

        return HuggingFaceInferenceAPI(
            model_name=HF_MODEL_NAME,
            token=HF_TOKEN,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    else:
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            request_timeout=REQUEST_TIMEOUT,
            additional_kwargs={
                "num_ctx": 2048,
                "num_predict": LLM_MAX_TOKENS,
            },
        )


def build_index_from_dir(data_dir):
    """
    Build a vector index from all documents in the given directory,
    then persist it to PERSIST_DIR.

    Clears any existing persisted index before building.
    Returns the newly built VectorStoreIndex.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise FileNotFoundError(
            f"No documents found in '{data_dir}/'. "
            "Please add files and try again."
        )

    # Clear old index
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    embed_model = get_embed_model()
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


def load_existing_index():
    """
    Load a previously persisted index from PERSIST_DIR.
    Raises FileNotFoundError if no index exists.
    """
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        raise FileNotFoundError(
            f"No persisted index found in '{PERSIST_DIR}/'. "
            "Upload documents or run: python build_index.py"
        )

    embed_model = get_embed_model()
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )


def create_query_engine(index):
    """Wrap a VectorStoreIndex into a query engine with the configured LLM."""
    llm = get_llm()
    return index.as_query_engine(
        llm=llm,
        response_mode="compact",
        similarity_top_k=SIMILARITY_TOP_K,
    )


# ── Convenience ──────────────────────────────────────────────────────────────


def load_rag_engine():
    """
    Convenience function: load existing index (or build from DATA_DIR),
    then return a ready-to-use query engine.
    """
    try:
        index = load_existing_index()
    except FileNotFoundError:
        index = build_index_from_dir(DATA_DIR)

    return create_query_engine(index)