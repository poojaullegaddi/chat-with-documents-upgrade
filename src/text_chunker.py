from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts(documents):
    """
    Split page-level Documents into smaller Documents.

    split_documents() carries source/page metadata into each resulting chunk.
    add_start_index adds the character offset of a chunk within its source page.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    return text_splitter.split_documents(documents)
