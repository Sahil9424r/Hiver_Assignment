"""
Tests for ChromaDB Vector Store retrieval and indexing.
"""

import pytest
from src.vector_store import ChromaResolutionStore, get_vector_store


def test_vector_store_initialization():
    store = get_vector_store()
    assert store is not None
    assert store.collection is not None
    assert store.count() > 0


def test_semantic_retrieval():
    store = get_vector_store()
    query = "My battery drops quickly and dies after iOS update"
    hits = store.search_similar_resolutions(query, top_k=2)

    assert len(hits) > 0
    top_hit = hits[0]
    assert "matched_customer_query" in top_hit
    assert "support_reply" in top_hit
    assert "similarity_score" in top_hit
    assert top_hit["similarity_score"] > 0.0
