# 📚 Chat With Your Documents – RAG Application

An AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)**. Upload PDF documents and interact with their content using natural language. The application retrieves relevant document passages, filters unrelated content, and generates context-grounded answers using Large Language Models (LLMs).

Built with Python, LangChain, FAISS, Sentence Transformers, Groq and Streamlit.

---

## 🚀 Features

- **PDF Document Upload:** Upload one or multiple PDF documents through an interactive interface.
- **PDF Text Extraction:** Extract text and page-level metadata using PyPDFLoader.
- **Intelligent Text Chunking:** Split documents into smaller chunks using RecursiveCharacterTextSplitter with configurable chunk size and overlap.
- **Semantic Embeddings:** Generate vector representations using Sentence Transformers.
- **Vector Similarity Search:** Store and retrieve document embeddings using FAISS.
- **Conversational Question Answering:** Ask questions and follow up using conversation context.
- **Query Rewriting:** Convert context-dependent follow-up questions into standalone search queries.
- **Relevance Filtering:** Use an LLM-based relevance evaluation step to filter unrelated retrieved passages.
- **Context-Grounded Responses:** Generate answers using retrieved document content rather than relying solely on the model's general knowledge.
- **Source Attribution:** Display document names, page numbers and retrieved text snippets.
- **Persistent Chat History:** Store conversations using SQLite.
- **Knowledge Base Management:** Replace the active document collection with newly uploaded PDFs.
- **Interactive Interface:** Built using Streamlit.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Frontend and user interface |
| LangChain | RAG pipeline orchestration |
| Groq API | LLM inference |
| OpenAI GPT-OSS 20B | Default configured language model |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| PyPDFLoader | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| SQLite | Conversation history storage |

## 🏗️ System Architecture

The application follows a Retrieval-Augmented Generation architecture with separate document ingestion and question-answering workflows.

### 1. Document Ingestion Pipeline

1. **PDF Upload:** Users upload PDF files through Streamlit.
2. **Text Extraction:** PyPDFLoader extracts text and page metadata from uploaded documents.
3. **Text Chunking:** RecursiveCharacterTextSplitter divides extracted text into smaller overlapping chunks.
4. **Embedding Generation:** Sentence Transformers converts document chunks into numerical vectors.
5. **Vector Indexing:** FAISS creates a searchable vector index from the generated embeddings.
6. **Persistence:** The FAISS index and document store are saved locally for reuse.

### 2. Question Answering Pipeline

1. **User Query:** The user enters a natural-language question.
2. **Conversation Context:** Previous messages are used to understand follow-up questions.
3. **Query Rewriting:** Context-dependent questions are rewritten into standalone search queries.
4. **Semantic Retrieval:** FAISS retrieves the top-k most similar document chunks.
5. **Relevance Filtering:** Groq LLM evaluates retrieved passages and identifies content relevant to the question.
6. **Context Preparation:** Relevant document passages are combined into a context for generation.
7. **Answer Generation:** The LLM generates an answer using the retrieved document context.
8. **Source Attribution:** Relevant document names, page numbers and snippets are displayed.
9. **Conversation Storage:** User and assistant messages are persisted in SQLite.

## 🔄 RAG Workflow

```text
             PDF Documents
                   |
                   v
            PDF Text Extraction
                   |
                   v
              Text Chunking
                   |
                   v
          Embedding Generation
                   |
                   v
              FAISS Index
                   |
                   |
             User Question
                   |
                   v
          Conversation Context
                   |
                   v
             Query Rewriting
                   |
                   v
          FAISS Similarity Search
                   |
                   v
           Relevance Filtering
                   |
                   v
          Relevant Document Chunks
                   |
                   v
           Groq LLM Generation
                   |
                   v
            Grounded Answer
                   |
                   v
        Sources and Page References
                   |
                   v
          SQLite Chat History
```

## 📁 Project Structure

```text
Chat-with-your-own-documents/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   ├── embedding_store.py
│   ├── rag_chain.py
│   └── chat_history.py
│
└── data/
    ├── pdfs/
    ├── faiss_index/
    └── chat_history.sqlite
```

**Module responsibilities:**

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, PDF upload, ingestion workflow and chat interaction |
| `pdf_loader.py` | Extract text and metadata from PDF documents |
| `text_chunker.py` | Split extracted documents into overlapping chunks |
| `embedding_store.py` | Generate embeddings and create, save and load FAISS indexes |
| `rag_chain.py` | Query rewriting, retrieval, relevance filtering and answer generation |
| `chat_history.py` | Store, retrieve and clear conversation messages using SQLite |

The `data/` directory is generated locally and is excluded from Git version control.



