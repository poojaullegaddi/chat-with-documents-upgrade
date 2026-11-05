
import os
import re
from typing import Any

import streamlit as st
from langchain_groq import ChatGroq


@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Create a reusable Groq chat-model client.

    The API key is loaded from environment variables or
    Streamlit secrets.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file "
            "or Streamlit secrets."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    return ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0,
        max_retries=2,
    )


def _message_content(response: Any) -> str:
    """Normalize the content returned by the chat model."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


def _format_history(
    chat_history: list[dict],
    max_messages: int = 8
) -> str:
    """Format recent conversation history."""

    recent = chat_history[-max_messages:]
    lines = []

    for message in recent:
        role = message.get("role", "user").upper()
        content = message.get("content", "")

        if content:
            lines.append(f"{role}: {content}")

    return (
        "\n".join(lines)
        if lines
        else "No previous conversation."
    )


def _make_standalone_question(
    llm,
    question: str,
    history_text: str
) -> str:
    """
    Rewrite follow-up questions into standalone search queries.
    """

    if history_text == "No previous conversation.":
        return question

    prompt = f"""
You rewrite follow-up questions into standalone search queries.

Conversation so far:
{history_text}

Latest user question:
{question}

Rewrite the latest question so it can be understood and searched
without the conversation history.

Preserve the user's intent.
Do not answer the question.
Return only the rewritten question.
"""

    response = llm.invoke(prompt)
    rewritten = _message_content(response).strip()

    return rewritten or question


def _filter_relevant_documents(
    llm,
    question: str,
    documents: list
) -> list:
    """
    Ask the LLM to identify which retrieved document chunks
    actually contain information relevant to the question.

    The model must return document numbers or NONE.
    """

    if not documents:
        return []

    document_context = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        document_context.append(
            f"""
DOCUMENT {index}
Source: {source}
Page: {page}
Content:
{document.page_content}
"""
        )

    context = "\n".join(document_context)

    prompt = f"""
You are a document relevance evaluator for a RAG application.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}

TASK:
Identify which documents contain information that is genuinely
useful for answering the user's question.

Rules:
1. Select a document only if its content provides relevant evidence.
2. Do not select a document merely because it contains similar words.
3. Do not select unrelated documents.
4. A document containing the exact answer should be selected.
5. A document containing necessary supporting information should
   also be selected.
6. For summary questions, select documents that discuss the topic.
7. If none of the documents are relevant, return NONE.

OUTPUT FORMAT:
Return ONLY document numbers separated by commas.

Example:
1, 3

If no documents are relevant, return:
NONE

Do not provide explanations.
"""

    response = llm.invoke(prompt)
    result = _message_content(response).strip()

    # Accept only a valid list of document numbers or NONE.
    first_line = result.splitlines()[0].strip() if result else ""

    if first_line.upper() == "NONE":
        return []

    if not re.fullmatch(
        r"\d+(\s*,\s*\d+)*",
        first_line
    ):
        return []

    selected_numbers = {
        int(number.strip())
        for number in first_line.split(",")
    }

    relevant_documents = []

    for index, document in enumerate(documents, start=1):
        if index in selected_numbers:
            relevant_documents.append(document)

    return relevant_documents


def answer_question(
    llm,
    vector_store,
    question: str,
    chat_history: list[dict] | None = None,
    k: int = 4,
):
    """
    Retrieve relevant document chunks and generate a grounded answer.

    Returns:
        {
            "answer": "...",
            "sources": [...]
        }
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chat_history = chat_history or []

    history_text = _format_history(chat_history)

    # STEP 1: Rewrite follow-up questions.
    search_query = _make_standalone_question(
        llm,
        question,
        history_text
    )

    # STEP 2: Retrieve candidate chunks from FAISS.
    retrieved_docs = vector_store.similarity_search(
        search_query,
        k=k
    )

    if not retrieved_docs:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "uploaded documents to answer that question."
            ),
            "sources": [],
        }

    # STEP 3: Filter unrelated document chunks.
    relevant_docs = _filter_relevant_documents(
        llm,
        search_query,
        retrieved_docs
    )

    # STEP 4: If no relevant chunks exist, do not generate
    # an answer from unrelated documents.
    if not relevant_docs:
        return {
            "answer": (
                "I couldn't find relevant information about this "
                "question in the uploaded documents."
            ),
            "sources": [],
        }

    # STEP 5: Build context using only relevant documents.
    context_parts = []
    sources = []
    seen_sources = set()

    for index, document in enumerate(
        relevant_docs,
        start=1
    ):
        metadata = document.metadata or {}

        source_name = metadata.get(
            "source",
            "Unknown source"
        )

        page_number = metadata.get(
            "page",
            "Unknown"
        )

        snippet = document.page_content.strip()

        context_parts.append(
            f"""
[{index}]
Source: {source_name}
Page: {page_number}

{snippet}
"""
        )

        source_key = (
            str(source_name),
            str(page_number)
        )

        if source_key not in seen_sources:
            seen_sources.add(source_key)

            sources.append({
                "source": str(source_name),
                "page": str(page_number),
                "snippet": snippet[:700],
            })

    context = "\n".join(context_parts)

    # STEP 6: Generate an answer using relevant context only.
    answer_prompt = f"""
You are a careful assistant that answers questions about
the user's uploaded PDF documents.

RULES:
- Answer using only the DOCUMENT CONTEXT below.
- If the answer is not present, say that you could not find
  the information in the uploaded documents.
- Never invent facts, dates, numbers or explanations.
- Use conversation history only to understand references.
- Do not use conversation history as factual evidence.
- Cite supporting context identifiers such as [1] or [2].
- If sources disagree, explain the disagreement.
- Be clear and concise.

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{question}

STANDALONE SEARCH QUERY:
{search_query}

DOCUMENT CONTEXT:
{context}

Answer:
"""

    response = llm.invoke(answer_prompt)

    answer = _message_content(response).strip()

    return {
        "answer": answer,
        "sources": sources,
    }
