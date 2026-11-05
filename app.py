
import os
import re
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project's root .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

import streamlit as st


from src.chat_history import (
    clear_conversation,
    init_chat_db,
    load_conversation,
    save_message,
)
from src.embedding_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
)
from src.pdf_loader import load_pdfs
from src.rag_chain import answer_question, get_llm
from src.text_chunker import chunk_texts

# ---------------------------------------------------------------------
# Paths and application configuration
# ---------------------------------------------------------------------

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "faiss_index"
STAGING_PDF_DIR = DATA_DIR / "_staging_pdfs"
STAGING_INDEX_DIR = DATA_DIR / "_staging_faiss_index"
DB_PATH = DATA_DIR / "chat_history.sqlite"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat With Your Documents")
st.caption(
    "Ask questions about your PDFs. Answers are grounded in retrieved "
    "document passages and include source/page references."
)

# ---------------------------------------------------------------------
# Persistent browser conversation identity
# The query parameter lets the same browser URL reopen the same chat.
# This is a demo convenience, not authentication or access control.
# ---------------------------------------------------------------------
if "chat_session_id" not in st.session_state:
    existing_chat_id = st.query_params.get("chat_id")
    chat_id = str(existing_chat_id) if existing_chat_id else str(uuid.uuid4())
    st.session_state.chat_session_id = chat_id
    st.query_params["chat_id"] = chat_id

init_chat_db(DB_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = load_conversation(
        DB_PATH, st.session_state.chat_session_id
    )

# Load the persisted FAISS index once per Streamlit session.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store(INDEX_DIR)

# Initialize the LLM lazily, only when needed.
if "llm" not in st.session_state:
    st.session_state.llm = None


def safe_pdf_filename(filename: str) -> str:
    """Remove directory components and unsafe characters from an upload name."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith(".pdf"):
        raise ValueError(f"{filename!r} is not a PDF file.")
    return name or f"upload_{uuid.uuid4().hex}.pdf"


def replace_directory(staging_dir: Path, final_dir: Path) -> None:
    """Replace a directory only after the staged content has been created."""
    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging_dir.replace(final_dir)


def process_uploaded_pdfs(uploaded_files) -> None:
    """
    Replace the current knowledge base with the PDFs uploaded in this run.
    The existing index is not changed until extraction, chunking, embedding,
    and saving the new index have succeeded.
    """
    if not uploaded_files:
        st.warning("Upload at least one PDF before processing.")
        return

    # Remove leftovers from a previous interrupted attempt.
    for path in (STAGING_PDF_DIR, STAGING_INDEX_DIR):
        if path.exists():
            shutil.rmtree(path)

    STAGING_PDF_DIR.mkdir(parents=True, exist_ok=True)

    try:
        seen_names = set()
        for uploaded_file in uploaded_files:
            filename = safe_pdf_filename(uploaded_file.name)
            if filename in seen_names:
                raise ValueError(
                    f"Two uploaded files have the same filename: {filename}"
                )
            seen_names.add(filename)

            file_bytes = uploaded_file.getvalue()
            if not file_bytes.startswith(b"%PDF"):
                raise ValueError(
                    f"{filename} does not appear to be a valid PDF file."
                )
            if len(file_bytes) > 50 * 1024 * 1024:
                raise ValueError(f"{filename} is larger than the 50 MB limit.")

            (STAGING_PDF_DIR / filename).write_bytes(file_bytes)

        with st.spinner("Extracting text from PDFs..."):
            documents = load_pdfs(STAGING_PDF_DIR)

        if not documents:
            raise ValueError("No readable text was extracted from the PDFs.")

        with st.spinner("Splitting documents into chunks..."):
            chunks = chunk_texts(documents)

        if not chunks:
            raise ValueError("No text chunks were created from the uploaded PDFs.")

        with st.spinner("Generating embeddings and building the FAISS index..."):
            new_vector_store = create_vector_store(chunks)

        # Save the new index to a staging directory first.
        save_vector_store(new_vector_store, STAGING_INDEX_DIR)

        # Only replace the current files/index after the new index is ready.
        replace_directory(STAGING_PDF_DIR, PDF_DIR)
        replace_directory(STAGING_INDEX_DIR, INDEX_DIR)

        st.session_state.vector_store = new_vector_store

        # A new document collection should start with a clean conversation,
        # so old answers are not mixed with the new knowledge base.
        clear_conversation(DB_PATH, st.session_state.chat_session_id)
        st.session_state.messages = []

        st.success(
            f"Processed {len(uploaded_files)} PDF file(s), "
            f"{len(documents)} page(s), and {len(chunks)} chunk(s). "
            "The previous knowledge base was replaced."
        )

    except Exception as exc:
        # Keep the previous in-memory index if the new ingestion failed.
        if STAGING_PDF_DIR.exists():
            shutil.rmtree(STAGING_PDF_DIR, ignore_errors=True)
        if STAGING_INDEX_DIR.exists():
            shutil.rmtree(STAGING_INDEX_DIR, ignore_errors=True)
        st.error(f"PDF processing failed: {exc}")


# ---------------------------------------------------------------------
# Sidebar: document ingestion and knowledge-base status
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Document library")
    st.write(
        "Upload PDFs and click **Replace knowledge base**. "
        "This version intentionally replaces the previous document collection."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum size: 50 MB per PDF.",
    )

    if st.button("Replace knowledge base", use_container_width=True):
        process_uploaded_pdfs(uploaded_files)

    st.divider()
    if st.session_state.vector_store is not None:
        st.success("A persisted FAISS knowledge base is loaded.")
    else:
        st.info("No knowledge base found. Upload and process PDFs to begin.")

    st.caption("PDFs and the FAISS index are stored locally under the data/ folder.")

# ---------------------------------------------------------------------
# Chat history controls and display
# ---------------------------------------------------------------------
left, right = st.columns([5, 1])
with left:
    st.subheader("💬 Conversation")
with right:
    if st.button("Clear chat", use_container_width=True):
        clear_conversation(DB_PATH, st.session_state.chat_session_id)
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Retrieved sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**[{index}] {source['source']} — "
                        f"page {source['page']}**"
                    )
                    st.caption(source["snippet"])

# ---------------------------------------------------------------------
# User question and RAG execution
# ---------------------------------------------------------------------
question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs before asking questions.")
        st.stop()

    user_message = {"role": "user", "content": question, "sources": []}
    st.session_state.messages.append(user_message)
    save_message(DB_PATH, st.session_state.chat_session_id, user_message)

    with st.chat_message("user"):
        st.markdown(question)

    try:
        if st.session_state.llm is None:
            st.session_state.llm = get_llm()

        # Use previous turns (excluding the just-added current question)
        # to resolve follow-up questions such as "What about its exceptions?"
        previous_messages = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and generating an answer..."):
                result = answer_question(
                    llm=st.session_state.llm,
                    vector_store=st.session_state.vector_store,
                    question=question,
                    chat_history=previous_messages,
                    k=4,
                )

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📄 Retrieved sources", expanded=True):
                    for index, source in enumerate(result["sources"], start=1):
                        st.markdown(
                            f"**[{index}] {source['source']} — "
                            f"page {source['page']}**"
                        )
                        st.caption(source["snippet"])

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
        st.session_state.messages.append(assistant_message)
        save_message(DB_PATH, st.session_state.chat_session_id, assistant_message)

    except Exception as exc:
        st.error(f"Unable to answer this question: {exc}")
