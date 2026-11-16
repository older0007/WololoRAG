"""
Token-aware text chunker (src/core/vector/chunker.py).

Why we chunk by TOKENS (not raw characters):
    LLMs and embedding models have strict token limits and think in BPE tokens
    (pieces of words). Measuring chunk sizes in tokens via `tiktoken` guarantees
    consistent semantic density and respects `config/vector_config.yaml`.
"""

from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import VectorConfig
from src.core.db.models import ArticleRecord


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the exact number of BPE tokens in `text` using tiktoken."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_article_into_chunks(
    article: ArticleRecord,
    vector_config: VectorConfig,
) -> list[Document]:
    """
    Convert an `ArticleRecord` from SQLite into 1 or more LangChain `Document` chunks.

    Behavior driven by `config/vector_config.yaml`:
    - If `chunking.enabled == False`: returns a single Document with the full article.
    - If `chunking.enabled == True`: uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder`
      to split cleanly along paragraph -> newline -> sentence -> word boundaries
      while strictly obeying `chunk_size_tokens` and `chunk_overlap_tokens`.
    """
    body = (article.cleaned_content or "").strip()
    if not body:
        raise ValueError(
            f"Article wp_post_id={article.wp_post_id} has empty cleaned_content; cannot chunk."
        )

    # Prepend title and topic header so semantic chunks retain article context.
    full_text = f"Title: {article.title}\nTopic: {article.topic}\n\n{body}"
    encoding_name = vector_config.chunking.tokenizer_encoding

    if not vector_config.chunking.enabled:
        raw_chunks = [full_text]
    else:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=vector_config.chunking.chunk_size_tokens,
            chunk_overlap=vector_config.chunking.chunk_overlap_tokens,
        )
        raw_chunks = splitter.split_text(full_text)

    if not raw_chunks:
        raw_chunks = [full_text]

    total_chunks = len(raw_chunks)
    published_iso = article.published_at.isoformat() if article.published_at else None

    documents: list[Document] = []
    for idx, chunk_text in enumerate(raw_chunks):
        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "wp_post_id": article.wp_post_id,
                    "title": article.title,
                    "url": article.url,
                    "topic": article.topic,
                    "published_at": published_iso,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "token_count": count_tokens(chunk_text, encoding_name=encoding_name),
                },
            )
        )

    return documents
