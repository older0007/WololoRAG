"""
Unit & Integration Tests for WololoRAG (Indexer Worker).

Verifies:
1. Pydantic validation (`app_config.yaml`, `vector_config.yaml`, Mistral API key guard, DATABASE_URL override).
2. Token-based chunking (`enabled=True` splits into bounded token chunks; `enabled=False` yields 1 chunk).
3. SQLite/SQLAlchemy ArticleRepository lifecycle (`DISCOVERED -> INDEXED`, `OUTDATED` detection, `FAILED` + `error_str`).
4. Full LangGraph Ingestion & Retrieval StateGraphs using a deterministic fake `Embeddings` provider.
5. Local Qdrant non-blocking access across multiple service instances.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import (
    AppConfig,
    ChunkingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    FullSettings,
    VectorConfig,
)
from src.core.db.models import ArticleStatus
from src.core.db.repository import ArticleRepository
from src.core.graphs.ingestion import run_ingestion_batch
from src.core.graphs.retrieval import run_retrieval_query
from src.core.vector.chunker import count_tokens, split_article_into_chunks
from src.core.vector.qdrant_store import QdrantStoreManager
from src.services.indexer.wololo_client import WololoClient


class DeterministicFakeEmbeddings(Embeddings):
    """
    Lightweight 8-dimensional deterministic embedding mock for fast offline tests.
    """

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [
            float(lowered.count("ps5") + 0.1),
            float(lowered.count("exploit") + lowered.count("jailbreak") + 0.1),
            float(lowered.count("switch") + lowered.count("nintendo") + 0.1),
            float(lowered.count("vita") + lowered.count("psp") + 0.1),
            float(lowered.count("emulator") + 0.1),
            float(lowered.count("homebrew") + 0.1),
            float(lowered.count("firmware") + 0.1),
            0.2,
        ]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def _make_test_settings(tmp_path: Path, chunk_enabled: bool = True) -> FullSettings:
    return FullSettings(
        app=AppConfig(
            embedding={"provider": "local"},
            database={
                "sqlite_path": str(tmp_path / "test_state.db"),
                "qdrant_mode": "local",
                "qdrant_local_path": str(tmp_path / "qdrant"),
                "collection_prefix": "test_wololo",
            },
        ),
        vector=VectorConfig(
            chunking={
                "enabled": chunk_enabled,
                "chunk_size_tokens": 40,
                "chunk_overlap_tokens": 10,
                "tokenizer_encoding": "cl100k_base",
            },
            search={
                "top_k": 3,
                "score_threshold": 0.1,
                "distance_metric": "Cosine",
            },
        ),
    )


def test_pydantic_mistral_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        EmbeddingConfig(provider="mistral")


def test_pydantic_database_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    cfg = DatabaseConfig()
    assert cfg.sqlalchemy_url == "postgresql+psycopg://u:p@localhost:5432/db"
    assert cfg.qdrant_mode == "server"
    assert cfg.qdrant_url == "http://localhost:6333"


def test_pydantic_chunk_overlap_validation() -> None:
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        ChunkingConfig(enabled=True, chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_chunker_toggle_on_vs_off(tmp_path: Path) -> None:
    repo = ArticleRepository(tmp_path / "db.sqlite")
    long_body = " ".join(["PS5 homebrew kernel exploit discussion paragraph."] * 40)
    repo.upsert_discovered_articles(
        [
            {
                "wp_post_id": 101,
                "title": "PS5 Exploit News",
                "url": "https://wololo.net/101",
                "topic": "PS5",
                "cleaned_content": long_body,
            }
        ]
    )
    article = repo.get_by_ids([101])[101]

    vec_on = _make_test_settings(tmp_path, chunk_enabled=True).vector
    chunks_on = split_article_into_chunks(article, vec_on)
    assert len(chunks_on) > 1
    for doc in chunks_on:
        assert doc.metadata["wp_post_id"] == 101
        assert count_tokens(doc.page_content) <= vec_on.chunking.chunk_size_tokens + 5

    vec_off = _make_test_settings(tmp_path, chunk_enabled=False).vector
    chunks_off = split_article_into_chunks(article, vec_off)
    assert len(chunks_off) == 1
    assert chunks_off[0].metadata["total_chunks"] == 1


def test_langgraph_two_phase_ingestion_error_str_and_retrieval(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, chunk_enabled=True)
    repo = ArticleRepository(settings.app.database.sqlalchemy_url)
    qdrant = QdrantStoreManager(settings.app, settings.vector, "test_wololo_local")
    embeddings = DeterministicFakeEmbeddings()

    try:
        summary = repo.upsert_discovered_articles(
            [
                {
                    "wp_post_id": 1,
                    "title": "PS5 Firmware 9.00 Exploit Released",
                    "url": "https://wololo.net/ps5-exploit",
                    "topic": "PS5",
                    "published_at": datetime(2026, 1, 10, tzinfo=timezone.utc),
                    "remote_modified_at": datetime(2026, 1, 10, tzinfo=timezone.utc),
                    "cleaned_content": (
                        "A new PS5 kernel exploit targeting firmware 9.00 has been demonstrated "
                        "by homebrew developers for research purposes."
                    ),
                },
                {
                    "wp_post_id": 2,
                    "title": "Nintendo Switch Emulator Update",
                    "url": "https://wololo.net/switch-emu",
                    "topic": "Switch",
                    "published_at": datetime(2026, 1, 11, tzinfo=timezone.utc),
                    "remote_modified_at": datetime(2026, 1, 11, tzinfo=timezone.utc),
                    "cleaned_content": (
                        "Nintendo Switch emulator performance improves with better shader caching."
                    ),
                },
                {
                    "wp_post_id": 3,
                    "title": "Broken Empty Article",
                    "url": "https://wololo.net/empty",
                    "topic": "General",
                    "published_at": datetime(2026, 1, 12, tzinfo=timezone.utc),
                    "remote_modified_at": datetime(2026, 1, 12, tzinfo=timezone.utc),
                    "cleaned_content": "",
                },
            ]
        )
        assert summary["inserted"] == 3
        assert repo.get_stats("local").discovered_count == 3

        pending = repo.get_pending_for_vectorization(limit=3, active_provider="local")
        batch_result = run_ingestion_batch(pending, settings, repo, embeddings, qdrant)

        assert batch_result.processed == 3
        assert batch_result.succeeded == 2
        assert batch_result.failed == 1
        assert qdrant.count_vectors() >= 2

        stats_after = repo.get_stats("local")
        assert stats_after.indexed_for_active_provider == 2
        assert stats_after.failed_count == 1

        failed_row = repo.get_by_ids([3])[3]
        assert failed_row.status == ArticleStatus.FAILED.value
        assert failed_row.error_str is not None
        assert "empty text" in failed_row.error_str

        ps5_row = repo.get_by_ids([1])[1]
        assert ps5_row.status == ArticleStatus.INDEXED.value
        assert len(ps5_row.vector_ids) >= 1

        upd = repo.upsert_discovered_articles(
            [
                {
                    "wp_post_id": 1,
                    "title": "PS5 Firmware 9.00 Exploit Released (Updated)",
                    "url": "https://wololo.net/ps5-exploit",
                    "topic": "PS5",
                    "remote_modified_at": datetime(2026, 2, 1, tzinfo=timezone.utc),
                    "cleaned_content": "Updated PS5 exploit guide with new jailbreak steps.",
                }
            ]
        )
        assert upd["updated_to_outdated"] == 1
        assert repo.get_by_ids([1])[1].status == ArticleStatus.OUTDATED.value

        retrieval = run_retrieval_query(
            query="PS5 jailbreak exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant,
            topic_filter="PS5",
        )
        hits = retrieval["enriched_hits"]
        assert len(hits) >= 1
        assert hits[0]["wp_post_id"] == 1
        assert "https://wololo.net/ps5-exploit" in retrieval["formatted_context"]
    finally:
        qdrant.close()


def test_html_cleaner_strips_scripts() -> None:
    raw = (
        "<p>Hello <a href='https://wololo.net'><b>Wololo</b></a> world!</p>"
        "<script>alert('ad')</script>"
        "<p>Second paragraph.</p>"
    )
    cleaned = WololoClient.clean_html_to_text(raw)
    assert "alert" not in cleaned
    assert "Hello Wololo world!" in cleaned
    assert "Second paragraph." in cleaned


def test_local_qdrant_allows_two_services_without_idle_lock(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, chunk_enabled=True)
    indexer_store = QdrantStoreManager(settings.app, settings.vector, "shared_local")
    reader_store = QdrantStoreManager(settings.app, settings.vector, "shared_local")
    embeddings = DeterministicFakeEmbeddings()

    try:
        doc = Document(
            page_content="PS5 homebrew news",
            metadata={"wp_post_id": 500, "title": "PS5", "url": "https://wololo.net/500", "topic": "PS5"},
        )
        vec = embeddings.embed_documents([doc.page_content])
        ids = indexer_store.upsert_chunks([doc], vec)
        assert len(ids) == 1

        assert reader_store.count_vectors() == 1
        hits = reader_store.search(embeddings.embed_query("PS5 homebrew"), top_k=3, score_threshold=0.1)
        assert len(hits) == 1
        assert hits[0]["wp_post_id"] == 500
    finally:
        indexer_store.close()
        reader_store.close()


def test_deterministic_uuid5_idempotency_and_clear_databases(tmp_path: Path) -> None:
    """
    Verify:
    1. Re-upserting the same chunk produces identical deterministic UUID5 IDs (no duplicates).
    2. Re-ingesting a shorter updated article cleans up orphan chunks via delete_article_vectors.
    3. clear_collection() and clear_all_articles() cleanly wipe both DBs.
    """
    settings = _make_test_settings(tmp_path, chunk_enabled=True)
    repo = ArticleRepository(settings.app.database.sqlalchemy_url)
    qdrant = QdrantStoreManager(settings.app, settings.vector, "test_wololo_local")
    embeddings = DeterministicFakeEmbeddings()

    try:
        long_text = " ".join(["PS5 kernel exploit details and payload steps."] * 35)
        repo.upsert_discovered_articles(
            [
                {
                    "wp_post_id": 900,
                    "title": "Long PS5 Article",
                    "url": "https://wololo.net/900",
                    "topic": "PS5",
                    "published_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
                    "remote_modified_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
                    "cleaned_content": long_text,
                }
            ]
        )

        # First ingestion -> multiple chunks
        res1 = run_ingestion_batch(
            repo.get_pending_for_vectorization(5, "local"),
            settings,
            repo,
            embeddings,
            qdrant,
        )
        assert res1.total_chunks_written > 1
        first_ids = repo.get_by_ids([900])[900].vector_ids

        # Re-upsert identical chunks -> deterministic UUID5 yields identical IDs and count
        chunks = split_article_into_chunks(repo.get_by_ids([900])[900], settings.vector)
        vecs = embeddings.embed_documents([c.page_content for c in chunks])
        second_ids = qdrant.upsert_chunks(chunks, vecs)
        assert second_ids == first_ids
        assert qdrant.count_vectors() == len(first_ids)

        # Update article to much shorter text (1 chunk) -> old extra chunks must be deleted
        repo.upsert_discovered_articles(
            [
                {
                    "wp_post_id": 900,
                    "title": "Shortened PS5 Article",
                    "url": "https://wololo.net/900",
                    "topic": "PS5",
                    "remote_modified_at": datetime(2025, 7, 1, tzinfo=timezone.utc),
                    "cleaned_content": "Short updated summary.",
                }
            ]
        )
        res2 = run_ingestion_batch(
            repo.get_pending_for_vectorization(5, "local"),
            settings,
            repo,
            embeddings,
            qdrant,
        )
        assert res2.total_chunks_written == 1
        assert qdrant.count_vectors() == 1

        # Clear both databases
        cleared_vectors = qdrant.clear_collection()
        cleared_rows = repo.clear_all_articles()
        assert cleared_vectors == 1
        assert cleared_rows == 1
        assert qdrant.count_vectors() == 0
        assert repo.get_stats("local").total_in_sql == 0
    finally:
        qdrant.close()


def test_per_article_dedup_and_full_article_context_mode(tmp_path: Path) -> None:
    """
    Verify:
    1. With max_chunks_per_article=1, multiple chunks from Post #1 do not crowd out Post #2.
    2. context_mode="full_article" swaps in complete cleaned text from SQLite.
    3. published_after filters out older articles.
    """
    settings = _make_test_settings(tmp_path, chunk_enabled=True)
    settings.vector.search.max_chunks_per_article = 1
    settings.vector.search.top_k = 5

    repo = ArticleRepository(settings.app.database.sqlalchemy_url)
    qdrant = QdrantStoreManager(settings.app, settings.vector, "test_wololo_local")
    embeddings = DeterministicFakeEmbeddings()

    try:
        repo.upsert_discovered_articles(
            [
                {
                    "wp_post_id": 10,
                    "title": "Older Multi-Chunk PS5 Post",
                    "url": "https://wololo.net/10",
                    "topic": "PS5",
                    "published_at": datetime(2023, 5, 1, tzinfo=timezone.utc),
                    "cleaned_content": " ".join(["PS5 exploit research details."] * 40),
                },
                {
                    "wp_post_id": 20,
                    "title": "Newer PS5 Homebrew Post",
                    "url": "https://wololo.net/20",
                    "topic": "PS5",
                    "published_at": datetime(2026, 2, 1, tzinfo=timezone.utc),
                    "cleaned_content": "PS5 exploit loader update FULL_BODY_SENTINEL.",
                },
            ]
        )
        run_ingestion_batch(
            repo.get_pending_for_vectorization(10, "local"),
            settings,
            repo,
            embeddings,
            qdrant,
        )
        assert qdrant.count_vectors() >= 3

        # Deduplication ensures at most 1 hit per post ID even when Post #10 has multiple chunks
        res_dedup = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant,
        )
        returned_post_ids = [h["wp_post_id"] for h in res_dedup["enriched_hits"]]
        assert returned_post_ids.count(10) == 1
        assert returned_post_ids.count(20) == 1

        # Date filter excludes 2023 article when published_after="2025-01-01"
        res_recent = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant,
            published_after="2025-01-01",
        )
        assert [h["wp_post_id"] for h in res_recent["enriched_hits"]] == [20]

        # context_mode="full_article" returns complete parent article from SQL
        settings.vector.search.context_mode = "full_article"
        res_full = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant,
            published_after="2025-01-01",
        )
        assert "[Mode: full_article" in res_full["formatted_context"]
        assert "FULL_BODY_SENTINEL" in res_full["formatted_context"]
    finally:
        qdrant.close()
