from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_dir: str | Path):
    """
    Load every PDF in a directory.

    Returns LangChain Document objects. Each page keeps metadata containing
    the original filename and a human-friendly 1-based page number.
    """
    pdf_dir = Path(pdf_dir)
    documents = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        page_documents = loader.load()

        for document in page_documents:
            # PyPDFLoader's page metadata is normally zero-based.
            zero_based_page = document.metadata.get("page", 0)
            document.metadata["source"] = pdf_path.name
            document.metadata["page"] = int(zero_based_page) + 1

        documents.extend(page_documents)

    return documents
