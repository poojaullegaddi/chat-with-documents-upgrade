import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource(show_spinner="Loading the sentence embedding model...")
def get_embedding_model():
    """
    Load one embedding model per Streamlit server process.

    The same model must be used for indexing documents and embedding queries.
    """
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(chunks):
    """
    Create a new FAISS index from LangChain Documents.

    from_documents() preserves each chunk's metadata, including source,
    page number, and start_index.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from zero chunks.")

    return FAISS.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
    )


def save_vector_store(vector_store, index_dir: str | Path):
    """Persist the FAISS index and its document store to disk."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))


def load_vector_store(index_dir: str | Path):
    """
    Load a previously saved FAISS index.

    allow_dangerous_deserialization=True is required by LangChain for its
    locally serialized docstore. Only load an index created by this app and
    stored in a directory you control. Never load an untrusted index file.
    """
    index_dir = Path(index_dir)

    index_file = index_dir / "index.faiss"
    docstore_file = index_dir / "index.pkl"

    if not index_file.exists() or not docstore_file.exists():
        return None

    return FAISS.load_local(
        str(index_dir),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
