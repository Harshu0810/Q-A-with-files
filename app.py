"""
Streamlit UI for the RAG Document Chatbot.
Users can upload files directly or import from Google Drive links.
Supports both local (Ollama) and HuggingFace Spaces deployment.
"""

import os
import shutil
import streamlit as st
from config import (
    IS_HF_SPACE,
    HF_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    EMBED_MODEL_NAME,
    SIMILARITY_TOP_K,
    PERSIST_DIR,
    UPLOAD_DIR,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="centered",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _prepare_upload_dir():
    """Clean and recreate the upload directory."""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def _build_and_activate(source_dir, doc_count):
    """Build index from a directory and activate the query engine."""
    from rag_engine import build_index_from_dir, create_query_engine

    index = build_index_from_dir(source_dir)
    st.session_state.query_engine = create_query_engine(index)
    st.session_state.doc_count = doc_count
    st.session_state.messages = []  # Clear old chat


def _download_from_gdrive(url, output_dir):
    """
    Download file or folder from a Google Drive shareable link.
    Returns the number of files downloaded.
    """
    import gdown

    if "/folders/" in url:
        gdown.download_folder(url, output=output_dir, quiet=True)
    else:
        # Trailing os.sep tells gdown to save with the original filename
        gdown.download(url, output=output_dir + os.sep, fuzzy=True, quiet=True)

    # Count downloaded files (recursive)
    count = 0
    for root, dirs, files in os.walk(output_dir):
        count += len([f for f in files if not f.startswith(".")])
    return count


# ── Session State Defaults ────────────────────────────────────────────────────
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ── Auto-load existing index on first run ─────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            from rag_engine import load_existing_index, create_query_engine

            with st.spinner("🔄 Loading existing index..."):
                index = load_existing_index()
                st.session_state.query_engine = create_query_engine(index)
        except Exception as e:
            st.warning(f"Could not load existing index: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Add Documents")

    tab_upload, tab_gdrive = st.tabs(["📤 Upload Files", "🔗 Google Drive"])

    # ── Tab 1: Direct file upload ─────────────────────────────────────────────
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Choose your files",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Upload one or more documents to build the search index.",
        )

        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) selected")
            for f in uploaded_files:
                st.caption(f"  • {f.name}")

            if st.button(
                "🔨 Build Index",
                key="btn_upload",
                type="primary",
                use_container_width=True,
            ):
                _prepare_upload_dir()
                for f in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, f.name)
                    with open(file_path, "wb") as out:
                        out.write(f.getbuffer())

                with st.spinner("🔨 Building search index..."):
                    try:
                        _build_and_activate(UPLOAD_DIR, len(uploaded_files))
                        st.success("✅ Index built!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to build index: {e}")

    # ── Tab 2: Google Drive link ──────────────────────────────────────────────
    with tab_gdrive:
        gdrive_url = st.text_input(
            "Paste a Google Drive link",
            placeholder="https://drive.google.com/file/d/... or .../folders/...",
        )
        st.caption(
            "Supports **files** and **folders**.  \n"
            "⚠️ Must be shared as *'Anyone with the link'*."
        )

        if gdrive_url:
            is_valid = "drive.google.com" in gdrive_url
            if not is_valid:
                st.error("Please enter a valid Google Drive URL.")
            else:
                is_folder = "/folders/" in gdrive_url
                label = "📥 Download Folder & Index" if is_folder else "📥 Download & Index"

                if st.button(
                    label,
                    key="btn_gdrive",
                    type="primary",
                    use_container_width=True,
                ):
                    _prepare_upload_dir()

                    with st.spinner("⬇️ Downloading from Google Drive..."):
                        try:
                            file_count = _download_from_gdrive(gdrive_url, UPLOAD_DIR)
                        except Exception as e:
                            st.error(
                                f"❌ Download failed: {e}\n\n"
                                "Make sure the file/folder is shared as "
                                "*'Anyone with the link'*."
                            )
                            file_count = 0

                    if file_count > 0:
                        with st.spinner(
                            f"🔨 Building index from {file_count} file(s)..."
                        ):
                            try:
                                _build_and_activate(UPLOAD_DIR, file_count)
                                st.success(f"✅ Indexed {file_count} file(s)!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to build index: {e}")
                    elif file_count == 0:
                        st.warning(
                            "No files were downloaded. Check that the link "
                            "is correct and publicly shared."
                        )

    # ── System Info ───────────────────────────────────────────────────────────
    st.divider()
    st.header("⚙️ System Info")

    if st.session_state.query_engine:
        st.success("✅ Ready to chat")
        if st.session_state.doc_count:
            st.caption(f"📄 {st.session_state.doc_count} document(s) indexed")
    else:
        st.warning("⏳ No index loaded")

    if IS_HF_SPACE:
        st.caption(f"☁️ **HF Spaces** — {HF_MODEL_NAME}")
    else:
        st.caption(f"🖥️ **Local** — {OLLAMA_MODEL_NAME}")

    st.caption(f"🔢 Embeddings: {EMBED_MODEL_NAME}")
    st.caption(f"🎯 Top-K: {SIMILARITY_TOP_K}")

    st.divider()
    st.markdown(
        "Built with [LlamaIndex](https://www.llamaindex.ai/) "
        "& [Streamlit](https://streamlit.io/)"
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📄 Document Question Answering")

if st.session_state.query_engine is None:
    # ── No index loaded — show prompt ─────────────────────────────────────────
    st.info(
        "👈 **Add your documents** in the sidebar to get started.\n\n"
        "You can **upload files** directly or paste a **Google Drive link** "
        "(file or folder).\n\n"
        "Supported formats: **PDF**, **TXT**, **Markdown**, **CSV**"
    )
    st.stop()

# ── Chat Interface (only shown when index is loaded) ─────────────────────────
st.write("Ask questions about your documents — powered by open-source AI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.write(f"**Page {src['page']}:**")
                    st.write(src["excerpt"])

# ── Chat Input ────────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                response = st.session_state.query_engine.query(user_question)
                st.markdown(str(response))

                # Extract and display source citations
                sources = []
                if hasattr(response, "source_nodes") and response.source_nodes:
                    with st.expander("📚 Sources"):
                        for node in response.source_nodes:
                            page_label = node.metadata.get("page_label", "?")
                            content = node.get_content()
                            excerpt = (
                                content[:300] + "..."
                                if len(content) > 300
                                else content
                            )
                            st.write(f"**Page {page_label}:**")
                            st.write(excerpt)
                            sources.append(
                                {"page": page_label, "excerpt": excerpt}
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(response),
                        "sources": sources,
                    }
                )
            except ConnectionError:
                if IS_HF_SPACE:
                    st.error(
                        "⚠️ Lost connection to HuggingFace Inference API. "
                        "Please try again."
                    )
                else:
                    st.error(
                        "⚠️ Lost connection to Ollama. "
                        "Make sure it's running: `ollama serve`"
                    )
            except Exception as e:
                st.error(f"⚠️ Error generating answer: {e}")