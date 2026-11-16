"""
Vector layer package: embeddings factory, token-based chunker, and Qdrant manager.
"""

from src.core.vector.chunker import count_tokens, split_article_into_chunks
from src.core.vector.embeddings import (
    create_embedding_model,
    get_collection_name,
    probe_embedding_dimension,
)
from src.core.vector.qdrant_store import QdrantStoreManager

__all__ = [
    "QdrantStoreManager",
    "count_tokens",
    "create_embedding_model",
    "get_collection_name",
    "probe_embedding_dimension",
    "split_article_into_chunks",
]
