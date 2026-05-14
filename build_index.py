"""
Build (or rebuild) the vector search index from documents in the data/ folder.
Run this script once after adding or updating your PDF files.

Usage:
    python build_index.py
"""

import os
import sys
from config import DATA_DIR, EMBED_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP


def build_index():
    # Validate data directory
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}/' not found.")
        print("   Create it and add your PDF files, then run this script again.")
        sys.exit(1)

    files = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]
    if not files:
        print(f"❌ No files found in '{DATA_DIR}/'.")
        print("   Add your PDF files to the data/ folder and run this script again.")
        sys.exit(1)

    print(f"📂 Found {len(files)} file(s) in '{DATA_DIR}/':")
    for f in files:
        print(f"   • {f}")

    print(f"\n🔢 Embedding model: {EMBED_MODEL_NAME}")
    print(f"📐 Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print("\n🔨 Building index (this may take a moment)...")

    from rag_engine import build_index_from_dir
    build_index_from_dir(DATA_DIR)

    print("\n✅ Index built successfully!")


if __name__ == "__main__":
    build_index()