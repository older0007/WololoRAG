"""
Qdrant Vector Database manager (src/core/vector/qdrant_store.py).

Stores dense vector embeddings alongside JSON metadata payloads (`title`, `topic`,
`url`, `text`) and performs fast nearest-neighbor similarity search with:
- Deterministic UUID5 point IDs (`collection:wp_post_id:chunk_index`) + filter deletion
  to keep SQL <-> Qdrant dual writes idempotent.
- Hybrid lexical + vector score blending, optional date filtering, and per-article
  chunk deduplication (`max_chunks_per_article`).

Concurrent Access Architecture:
    - `qdrant_mode: "local"` stores vectors directly on disk in `./data/qdrant`.
      We open `QdrantClient(path=...)` inside a short-lived context manager
      (`_client_session`) per operation with retry on lock contention so `indexer`
      and `mcp-server` do not block each other while idle.
    - `qdrant_mode: "server"` keeps a persistent HTTP client connected to Qdrant running
      in Docker (`http://localhost:6333`).
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document
from portalocker.exceptions import AlreadyLocked
from qdrant_client import QdrantClient, models

from src.core.config import AppConfig, VectorConfig


_DISTANCE_MAP: dict[str, models.Distance] = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
    "Euclid": models.Distance.EUCLID,
}

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "what", "when", "where", "which", "who", "with",
    }
)


def _tokenize_for_lexical(text: str) -> set[str]:
    """Extract lowercase alphanumeric query terms, dropping trivial stop words."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and (len(t) >= 2 or t.isdigit())}


def _parse_iso_utc(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    raw = dt_str.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class QdrantStoreManager:
    """Encapsulates collection lifecycle, upsert, delete, and hybrid semantic search in Qdrant."""

    def __init__(
        self,
        app_config: AppConfig,
        vector_config: VectorConfig,
        collection_name: str,
        lock_retry_timeout_sec: float = 5.0,
    ) -> None:
        self.app_config = app_config
        self.vector_config = vector_config
        self.collection_name = collection_name
        self.lock_retry_timeout_sec = lock_retry_timeout_sec

        self._thread_lock = threading.Lock()
        self._server_client: QdrantClient | None = None
        self._local_path: Path | None = None

        db_cfg = app_config.database
        if db_cfg.qdrant_mode == "local":
            local_dir = Path(db_cfg.qdrant_local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            self._local_path = local_dir
        elif db_cfg.qdrant_mode == "server":
            self._server_client = QdrantClient(url=db_cfg.qdrant_url)
        else:
            raise ValueError(f"Unknown qdrant_mode: {db_cfg.qdrant_mode!r}")

    @contextmanager
    def _client_session(self) -> Iterator[QdrantClient]:
        if self._server_client is not None:
            yield self._server_client
            return

        assert self._local_path is not None
        with self._thread_lock:
            deadline = time.monotonic() + self.lock_retry_timeout_sec
            client: QdrantClient | None = None

            while True:
                try:
                    client = QdrantClient(path=str(self._local_path))
                    break
                except (AlreadyLocked, RuntimeError) as exc:
                    is_lock_error = isinstance(exc, AlreadyLocked) or (
                        "already accessed by another instance" in str(exc)
                    )
                    if not is_lock_error or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)

            try:
                yield client
            finally:
                client.close()

    def close(self) -> None:
        """Close the persistent Qdrant HTTP client if running in 'server' mode."""
        if self._server_client is not None:
            self._server_client.close()
            self._server_client = None

    def _ensure_collection_with_client(
        self,
        client: QdrantClient,
        vector_dimension: int,
    ) -> None:
        """Create or verify collection dimension on an open client."""
        if client.collection_exists(self.collection_name):
            info = client.get_collection(self.collection_name)
            existing_params = info.config.params.vectors
            if isinstance(existing_params, models.VectorParams):
                if existing_params.size != vector_dimension:
                    raise ValueError(
                        f"Qdrant collection {self.collection_name!r} expects dimension "
                        f"{existing_params.size}, but active embedding model produces "
                        f"{vector_dimension}. Check collection_prefix or embedding provider."
                    )
            return

        metric_name = self.vector_config.search.distance_metric
        distance = _DISTANCE_MAP[metric_name]

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimension,
                distance=distance,
            ),
        )

    def ensure_collection(self, vector_dimension: int) -> None:
        """Create the Qdrant collection if absent or verify its vector dimension."""
        with self._client_session() as client:
            self._ensure_collection_with_client(client, vector_dimension)

    def count_vectors(self) -> int:
        """Return total number of stored chunk vectors in the active collection."""
        with self._client_session() as client:
            if not client.collection_exists(self.collection_name):
                return 0
            return client.count(collection_name=self.collection_name, exact=True).count

    def delete_vectors(self, point_ids: list[str]) -> None:
        """Delete specific chunk vectors by their UUIDs."""
        if not point_ids:
            return
        with self._client_session() as client:
            if not client.collection_exists(self.collection_name):
                return
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=list(point_ids)),
            )

    def delete_article_vectors(self, wp_post_id: int) -> None:
        """
        Delete ALL vectors belonging to `wp_post_id` via metadata filter.
        Guarantees idempotent cleanup even if a previous run crashed mid-write
        before saving `vector_ids` into SQL.
        """
        with self._client_session() as client:
            if not client.collection_exists(self.collection_name):
                return
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="wp_post_id",
                                match=models.MatchValue(value=int(wp_post_id)),
                            )
                        ]
                    )
                ),
            )

    def clear_collection(self) -> int:
        """Drop the active Qdrant collection completely and return deleted vector count."""
        with self._client_session() as client:
            if not client.collection_exists(self.collection_name):
                return 0
            existing_count = client.count(collection_name=self.collection_name, exact=True).count
            client.delete_collection(self.collection_name)
            return existing_count

    def upsert_chunks(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
    ) -> list[str]:
        """
        Store chunk vectors + metadata payloads into Qdrant using deterministic UUID5 IDs.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match vector count ({len(vectors)})."
            )
        if not chunks:
            return []

        points: list[models.PointStruct] = []
        generated_ids: list[str] = []

        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            wp_post_id = chunk.metadata.get("wp_post_id")
            chunk_idx = chunk.metadata.get("chunk_index", idx)
            if wp_post_id is not None:
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{self.collection_name}:{wp_post_id}:{chunk_idx}",
                    )
                )
            else:
                point_id = str(uuid.uuid4())

            generated_ids.append(point_id)
            payload: dict[str, Any] = {
                "text": chunk.page_content,
                **chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        with self._client_session() as client:
            self._ensure_collection_with_client(client, vector_dimension=len(vectors[0]))
            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        return generated_ids

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
        topic_filter: str | None = None,
        query_text: str = "",
        published_after: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute semantic search with optional hybrid keyword boosting, date filtering,
        and per-article chunk deduplication (`max_chunks_per_article`).
        """
        search_cfg = self.vector_config.search
        effective_top_k = top_k if top_k is not None else search_cfg.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else search_cfg.score_threshold
        )
        kw_weight = search_cfg.hybrid_keyword_weight
        max_per_article = search_cfg.max_chunks_per_article

        query_tokens = _tokenize_for_lexical(query_text) if query_text.strip() else set()
        use_lexical = bool(query_tokens and kw_weight > 0.0)

        after_dt = _parse_iso_utc(published_after)

        # Widen candidate pool so post-filtering (dedup + date + hybrid re-rank) has plenty of hits
        candidate_limit = max(effective_top_k * 5, 25)
        if after_dt is not None:
            candidate_limit = max(candidate_limit, 60)

        # Lower pre-filter threshold slightly when lexical boost is active so strong keyword hits aren't dropped
        qdrant_threshold = (
            max(0.0, effective_threshold * (1.0 - kw_weight))
            if use_lexical
            else effective_threshold
        )

        qdrant_filter: models.Filter | None = None
        if topic_filter and topic_filter.strip():
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchValue(value=topic_filter.strip()),
                    )
                ]
            )

        with self._client_session() as client:
            if not client.collection_exists(self.collection_name):
                return []

            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=candidate_limit,
                score_threshold=qdrant_threshold,
                with_payload=True,
            )

        candidates: list[dict[str, Any]] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            if after_dt is not None:
                pub_dt = _parse_iso_utc(payload.get("published_at"))
                if pub_dt is None or pub_dt < after_dt:
                    continue

            vector_score = float(hit.score)
            lexical_score = 0.0
            if use_lexical:
                haystack = f"{payload.get('title', '')} {payload.get('text', '')}".lower()
                matched_terms = sum(1 for tok in query_tokens if tok in haystack)
                lexical_score = matched_terms / len(query_tokens)
                blended_score = (1.0 - kw_weight) * vector_score + kw_weight * lexical_score
            else:
                blended_score = vector_score

            if blended_score < effective_threshold:
                continue

            candidates.append(
                {
                    "vector_id": str(hit.id),
                    "score": blended_score,
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "text": payload.get("text", ""),
                    "wp_post_id": payload.get("wp_post_id"),
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "topic": payload.get("topic", "General"),
                    "published_at": payload.get("published_at"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 1),
                    "token_count": payload.get("token_count", 0),
                }
            )

        # Sort by blended score descending and deduplicate per article
        candidates.sort(key=lambda item: item["score"], reverse=True)

        deduped: list[dict[str, Any]] = []
        per_article_counts: dict[Any, int] = {}
        for item in candidates:
            post_key = item.get("wp_post_id")
            if post_key is None:
                post_key = item["vector_id"]
            seen = per_article_counts.get(post_key, 0)
            if seen >= max_per_article:
                continue
            per_article_counts[post_key] = seen + 1
            deduped.append(item)
            if len(deduped) >= effective_top_k:
                break

        return deduped
