"""
LangGraph Ingestion StateGraph (src/core/graphs/ingestion.py).

Wires together discrete processing nodes and conditional edges operating over
the shared `ArticleIngestionState` dictionary.

Graph Topology:
    START
      -> validate_and_prepare_text
      -> [on error -> record_failure_in_sql -> END]
      -> [chunking.enabled == True  -> chunk_by_tokens]
      -> [chunking.enabled == False -> single_document_chunk]
      -> embed_and_store_qdrant
      -> [on error -> record_failure_in_sql -> END]
      -> commit_sql_status
      -> END
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph

from src.core.config import FullSettings
from src.core.db.models import ArticleRecord, ArticleStatus
from src.core.db.repository import ArticleRepository, compute_content_hash
from src.core.graphs.state import ArticleIngestionState
from src.core.vector.chunker import split_article_into_chunks
from src.core.vector.qdrant_store import QdrantStoreManager


@dataclass
class IngestionResult:
    """Summary returned after running a batch of articles through the Ingestion Graph."""

    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    total_chunks_written: int = 0
    errors: list[tuple[int, str, str]] = field(default_factory=list)


def build_ingestion_graph(
    settings: FullSettings,
    repo: ArticleRepository,
    embeddings: Embeddings,
    qdrant_store: QdrantStoreManager,
):
    """
    Build and compile the LangGraph `StateGraph` for vectorizing a single article.
    Dependencies (`repo`, `embeddings`, `qdrant_store`) are injected via closures.
    """
    active_provider = settings.app.embedding.provider

    # --------------------------------------------------------------------------
    # Node 1: Validate cleaned text and compute fingerprint hash
    # --------------------------------------------------------------------------
    def validate_and_prepare_text(state: ArticleIngestionState) -> dict:
        article = state["article"]
        try:
            text = (article.cleaned_content or "").strip()
            if not text:
                raise ValueError(
                    f"Article {article.wp_post_id} ({article.title!r}) has empty text."
                )
            content_hash = compute_content_hash(text)
            return {
                "active_provider": active_provider,
                "content_hash": content_hash,
                "chunking_enabled": settings.vector.chunking.enabled,
                "error_str": None,
            }
        except Exception as exc:
            return {
                "error_str": f"[validate_and_prepare_text] {type(exc).__name__}: {exc}",
            }

    # --------------------------------------------------------------------------
    # Conditional Router: Check for error or choose chunking strategy
    # --------------------------------------------------------------------------
    def route_after_validation(state: ArticleIngestionState) -> str:
        if state.get("error_str"):
            return "record_failure_in_sql"
        if state.get("chunking_enabled", True):
            return "chunk_by_tokens"
        return "single_document_chunk"

    # --------------------------------------------------------------------------
    # Node 2a: Split article into overlapping token chunks (chunking.enabled=True)
    # --------------------------------------------------------------------------
    def chunk_by_tokens(state: ArticleIngestionState) -> dict:
        try:
            chunks = split_article_into_chunks(state["article"], settings.vector)
            return {"chunks": chunks}
        except Exception as exc:
            return {
                "error_str": f"[chunk_by_tokens] {type(exc).__name__}: {exc}",
            }

    # --------------------------------------------------------------------------
    # Node 2b: Keep full article as 1 chunk (chunking.enabled=False)
    # --------------------------------------------------------------------------
    def single_document_chunk(state: ArticleIngestionState) -> dict:
        try:
            chunks = split_article_into_chunks(state["article"], settings.vector)
            return {"chunks": chunks}
        except Exception as exc:
            return {
                "error_str": f"[single_document_chunk] {type(exc).__name__}: {exc}",
            }

    def route_after_chunking(state: ArticleIngestionState) -> str:
        if state.get("error_str"):
            return "record_failure_in_sql"
        return "embed_and_store_qdrant"

    # --------------------------------------------------------------------------
    # Node 3: Generate embeddings (Local or Mistral API) and upsert into Qdrant
    # --------------------------------------------------------------------------
    def embed_and_store_qdrant(state: ArticleIngestionState) -> dict:
        article = state["article"]
        chunks = state["chunks"]
        try:
            # Step 3a: Delete ALL existing Qdrant vectors for this wp_post_id via filter.
            # This keeps SQL <-> Qdrant idempotent even if a prior run failed mid-write.
            qdrant_store.delete_article_vectors(article.wp_post_id)

            # Step 3b: Embed all chunk strings in a batch using LangChain's interface.
            texts = [doc.page_content for doc in chunks]
            vectors = embeddings.embed_documents(texts)

            # Step 3c: Upsert points into Qdrant and capture generated UUIDs.
            vector_ids = qdrant_store.upsert_chunks(chunks=chunks, vectors=vectors)
            return {
                "vectors": vectors,
                "vector_ids": vector_ids,
            }
        except Exception as exc:
            return {
                "error_str": f"[embed_and_store_qdrant] {type(exc).__name__}: {exc}",
            }

    def route_after_qdrant(state: ArticleIngestionState) -> str:
        if state.get("error_str"):
            return "record_failure_in_sql"
        return "commit_sql_status"

    # --------------------------------------------------------------------------
    # Node 4a: Happy path — persist vector_ids and INDEXED status in SQLite
    # --------------------------------------------------------------------------
    def commit_sql_status(state: ArticleIngestionState) -> dict:
        article = state["article"]
        vector_ids = state.get("vector_ids", [])
        try:
            repo.mark_article_indexed(
                wp_post_id=article.wp_post_id,
                vector_ids=vector_ids,
                chunk_count=len(vector_ids),
                provider=active_provider,
                content_hash=state["content_hash"],
            )
            return {"status": ArticleStatus.INDEXED.value, "error_str": None}
        except Exception as exc:
            err_msg = f"[commit_sql_status] {type(exc).__name__}: {exc}"
            repo.mark_article_failed(article.wp_post_id, err_msg)
            return {"status": ArticleStatus.FAILED.value, "error_str": err_msg}

    # --------------------------------------------------------------------------
    # Node 4b: Error path — persist detailed `error_str` in SQLite for inspection
    # --------------------------------------------------------------------------
    def record_failure_in_sql(state: ArticleIngestionState) -> dict:
        article = state["article"]
        err_msg = state.get("error_str") or "Unknown ingestion error"
        repo.mark_article_failed(article.wp_post_id, err_msg)
        return {"status": ArticleStatus.FAILED.value, "error_str": err_msg}

    # --------------------------------------------------------------------------
    # Wire up the StateGraph (Nodes + Edges + Conditional Routing)
    # --------------------------------------------------------------------------
    graph = StateGraph(ArticleIngestionState)

    graph.add_node("validate_and_prepare_text", validate_and_prepare_text)
    graph.add_node("chunk_by_tokens", chunk_by_tokens)
    graph.add_node("single_document_chunk", single_document_chunk)
    graph.add_node("embed_and_store_qdrant", embed_and_store_qdrant)
    graph.add_node("commit_sql_status", commit_sql_status)
    graph.add_node("record_failure_in_sql", record_failure_in_sql)

    graph.add_edge(START, "validate_and_prepare_text")
    graph.add_conditional_edges(
        "validate_and_prepare_text",
        route_after_validation,
        {
            "chunk_by_tokens": "chunk_by_tokens",
            "single_document_chunk": "single_document_chunk",
            "record_failure_in_sql": "record_failure_in_sql",
        },
    )
    graph.add_conditional_edges(
        "chunk_by_tokens",
        route_after_chunking,
        {
            "embed_and_store_qdrant": "embed_and_store_qdrant",
            "record_failure_in_sql": "record_failure_in_sql",
        },
    )
    graph.add_conditional_edges(
        "single_document_chunk",
        route_after_chunking,
        {
            "embed_and_store_qdrant": "embed_and_store_qdrant",
            "record_failure_in_sql": "record_failure_in_sql",
        },
    )
    graph.add_conditional_edges(
        "embed_and_store_qdrant",
        route_after_qdrant,
        {
            "commit_sql_status": "commit_sql_status",
            "record_failure_in_sql": "record_failure_in_sql",
        },
    )
    graph.add_edge("commit_sql_status", END)
    graph.add_edge("record_failure_in_sql", END)

    return graph.compile()


def run_ingestion_batch(
    articles: list[ArticleRecord],
    settings: FullSettings,
    repo: ArticleRepository,
    embeddings: Embeddings,
    qdrant_store: QdrantStoreManager,
    on_progress: Callable[[int, int, ArticleRecord, str, int], None] | None = None,
) -> IngestionResult:
    """
    Execute the compiled Ingestion `StateGraph` across a batch of SQLite articles.
    Each article is isolated: if one fails, its `error_str` is saved to SQLite
    and the remaining articles continue processing.
    """
    compiled_graph = build_ingestion_graph(settings, repo, embeddings, qdrant_store)
    result = IngestionResult()
    total = len(articles)

    for idx, article in enumerate(articles, start=1):
        result.processed += 1
        try:
            final_state = compiled_graph.invoke({"article": article})
            final_status = final_state.get("status", ArticleStatus.FAILED.value)
            chunk_count = len(final_state.get("vector_ids", []))
            err_str = final_state.get("error_str")
        except Exception as exc:
            # Safety net if LangGraph itself raises an unhandled error.
            final_status = ArticleStatus.FAILED.value
            chunk_count = 0
            err_str = f"[graph_invoke] {type(exc).__name__}: {exc}"
            repo.mark_article_failed(article.wp_post_id, err_str)

        if final_status == ArticleStatus.INDEXED.value:
            result.succeeded += 1
            result.total_chunks_written += chunk_count
        else:
            result.failed += 1
            result.errors.append((article.wp_post_id, article.title, err_str or "Unknown error"))

        if on_progress is not None:
            on_progress(idx, total, article, final_status, chunk_count)

    return result
