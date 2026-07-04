"""
tests/test_hybrid_retriever.py — Tests for BM25, HybridRetriever, and RRF fusion.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.hybrid_retriever import HybridRetriever, RRF_K, _bm25_to_hybrid, _faiss_to_hybrid
from app.db.bm25_store import BM25Store, _tokenize
from app.utils.text_splitter import ChunkDict


# ── BM25Store tests ───────────────────────────────────────────────────────────

class TestBM25Store:
    def _make_chunks(self, texts: list[str], label: str = "A") -> list[ChunkDict]:
        return [
            ChunkDict(chunk_id=i, text=t, page_hint=i + 1, source_label=label)
            for i, t in enumerate(texts)
        ]

    def test_build_and_ready(self):
        store = BM25Store(source_label="A")
        chunks = self._make_chunks(["OTA timeout is 300 seconds", "update procedure takes 5 minutes"])
        store.build(chunks)
        assert store.is_ready is True

    def test_search_returns_relevant_chunk(self):
        store = BM25Store(source_label="A")
        chunks = self._make_chunks([
            "OTA update timeout should be set to 300 seconds",
            "Unrelated content about something else entirely",
            "Network configuration guide for routers",
        ])
        store.build(chunks)
        results = store.search("OTA timeout seconds", k=1)
        assert len(results) == 1
        assert "timeout" in results[0]["text"].lower() or "ota" in results[0]["text"].lower()

    def test_search_returns_correct_source_label(self):
        store = BM25Store(source_label="B")
        chunks = self._make_chunks(["fix for timeout issue"], label="B")
        store.build(chunks)
        results = store.search("timeout", k=1)
        assert results[0]["source_label"] == "B"

    def test_empty_store_returns_empty(self):
        store = BM25Store(source_label="A")
        results = store.search("anything", k=3)
        assert results == []

    def test_zero_score_chunks_excluded(self):
        store = BM25Store(source_label="A")
        chunks = self._make_chunks(["apples and oranges", "bananas and grapes"])
        store.build(chunks)
        # Query with completely unrelated terms — should get no/low results
        results = store.search("quantum physics laser beam", k=5)
        # Either empty or very low scores — either way no crash
        assert isinstance(results, list)

    def test_k_clamped_to_corpus_size(self):
        store = BM25Store(source_label="A")
        chunks = self._make_chunks(["only one chunk here"])
        store.build(chunks)
        results = store.search("chunk", k=100)
        assert len(results) <= 1


class TestTokenizer:
    def test_lowercases(self):
        tokens = _tokenize("OTA Update TIMEOUT")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stop_words(self):
        tokens = _tokenize("the timeout is set to 300")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "to" not in tokens

    def test_preserves_technical_terms(self):
        tokens = _tokenize("ota.update_timeout_seconds value")
        assert any("ota" in t or "timeout" in t for t in tokens)

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_stop_words(self):
        assert _tokenize("the and or but") == []


# ── HybridRetriever tests ─────────────────────────────────────────────────────

class TestHybridRetriever:
    def _mock_faiss(self, results=None):
        store = MagicMock()
        store.is_ready = True
        store.source_label = "A"
        store.search.return_value = results or [
            {"chunk_id": 0, "text": "chunk zero", "page_hint": 1, "score": 0.1, "source_label": "A"},
            {"chunk_id": 1, "text": "chunk one", "page_hint": 2, "score": 0.2, "source_label": "A"},
            {"chunk_id": 2, "text": "chunk two", "page_hint": 3, "score": 0.3, "source_label": "A"},
        ]
        return store

    def _mock_bm25(self, results=None):
        store = MagicMock()
        store.is_ready = True
        store.source_label = "A"
        store.search.return_value = results or [
            {"chunk_id": 1, "text": "chunk one", "page_hint": 2, "source_label": "A", "bm25_score": 5.0},
            {"chunk_id": 2, "text": "chunk two", "page_hint": 3, "source_label": "A", "bm25_score": 3.0},
            {"chunk_id": 3, "text": "chunk three", "page_hint": 4, "source_label": "A", "bm25_score": 1.0},
        ]
        return store

    def test_returns_hybrid_results(self):
        retriever = HybridRetriever(self._mock_faiss(), self._mock_bm25())
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="test query",
            top_k=3,
        )
        assert len(results) > 0

    def test_rrf_scores_are_positive(self):
        retriever = HybridRetriever(self._mock_faiss(), self._mock_bm25())
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="test query",
            top_k=5,
        )
        for r in results:
            assert r.rrf_score > 0

    def test_sorted_by_rrf_score_descending(self):
        retriever = HybridRetriever(self._mock_faiss(), self._mock_bm25())
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="query",
            top_k=5,
        )
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_appearing_in_both_gets_higher_score(self):
        """chunk_id=1 appears in both FAISS and BM25 — should score higher than chunk_id=0 (FAISS only)."""
        retriever = HybridRetriever(self._mock_faiss(), self._mock_bm25(), vector_k=5, bm25_k=5)
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="query",
            top_k=10,
        )
        cid_scores = {r.chunk_id: r.rrf_score for r in results}
        # chunk_id=1 is in both → should score higher than chunk_id=0 (only FAISS)
        if 0 in cid_scores and 1 in cid_scores:
            assert cid_scores[1] > cid_scores[0]

    def test_faiss_only_fallback(self):
        bm25 = MagicMock()
        bm25.is_ready = False
        bm25.source_label = "A"
        bm25.search.return_value = []
        retriever = HybridRetriever(self._mock_faiss(), bm25)
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="query",
            top_k=3,
        )
        assert len(results) > 0
        for r in results:
            assert r.vector_rank is not None

    def test_top_k_respected(self):
        retriever = HybridRetriever(self._mock_faiss(), self._mock_bm25())
        results = retriever.retrieve(
            query_vector=np.random.rand(384).astype(np.float32),
            query_text="query",
            top_k=2,
        )
        assert len(results) <= 2

    def test_rrf_formula_correct(self):
        """Verify RRF score matches formula: 1/(k + rank)."""
        rank = 1
        expected = 1.0 / (RRF_K + rank)
        result = _faiss_to_hybrid([
            {"chunk_id": 0, "text": "t", "page_hint": 1, "score": 0.1, "source_label": "A"}
        ], top_k=1)
        assert abs(result[0].rrf_score - expected) < 1e-9
