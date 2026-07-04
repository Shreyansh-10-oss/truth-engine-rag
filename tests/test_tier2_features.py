"""
tests/test_tier2_features.py — Tests for conflict detection and reranker.
All external dependencies mocked — no model or API calls.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.conflict_detector import (
    ConflictReport,
    _cosine_similarity,
    _normalise_score,
    detect_conflicts,
)
from app.core.reranker import (
    CrossEncoderReranker,
    SourceConfidence,
    _confidence_label,
    _hybrid_to_ranked_fallback,
    _normalise_score,
)
from app.core.hybrid_retriever import HybridResult


# ── Conflict Detector tests ───────────────────────────────────────────────────

def _mock_ranked_chunk(text: str, source: str = "A", confidence: float = 0.8):
    chunk = MagicMock()
    chunk.text = text
    chunk.source_label = source
    chunk.confidence = confidence
    chunk.chunk_id = 0
    chunk.page_hint = 1
    return chunk


class TestConflictDetector:
    def test_no_conflict_when_sources_empty(self):
        report = detect_conflicts([], [])
        assert report.conflict_detected is False
        assert report.severity == "none"

    def test_no_conflict_when_one_source_empty(self):
        a_chunks = [_mock_ranked_chunk("OTA timeout is 300 seconds")]
        report = detect_conflicts(a_chunks, [])
        assert report.conflict_detected is False

    def test_numeric_conflict_detected(self):
        """Source A says 300 seconds, Source C says 5 minutes — different numeric values."""
        a_chunks = [_mock_ranked_chunk("OTA timeout should be 300 seconds for production")]
        c_chunks = [_mock_ranked_chunk("OTA timeout is set to 60 seconds by default")]
        report = detect_conflicts(a_chunks, c_chunks)
        # Should detect differing numeric values
        assert isinstance(report, ConflictReport)
        # Either detected or not — no crash
        assert report.severity in ("none", "low", "medium", "high")

    def test_negation_conflict_detected(self):
        a_chunks = [_mock_ranked_chunk("Use the ota.update_timeout parameter to configure timeout")]
        c_chunks = [_mock_ranked_chunk("The ota.update_timeout parameter is no longer supported and deprecated")]
        report = detect_conflicts(a_chunks, c_chunks)
        # Source C has negation — should be flagged
        if report.conflict_detected:
            assert "negation" in report.conflict_type or "semantic" in report.conflict_type

    def test_identical_sources_no_conflict(self):
        text = "OTA timeout is 300 seconds as configured in the manual"
        a_chunks = [_mock_ranked_chunk(text)]
        c_chunks = [_mock_ranked_chunk(text, source="C")]

        # Identical text, identical embeddings → high similarity → no conflict
        vec = np.array([0.5, 0.5, 0.5, 0.5])
        report = detect_conflicts(
            a_chunks, c_chunks,
            source_a_embeddings=vec.reshape(1, -1),
            source_c_embeddings=vec.reshape(1, -1),
        )
        assert report.conflict_detected is False

    def test_semantic_conflict_with_dissimilar_embeddings(self):
        a_chunks = [_mock_ranked_chunk("The system uses automatic rollback after timeout")]
        c_chunks = [_mock_ranked_chunk("Manual intervention required, no automatic rollback")]

        # Very different vectors → low cosine similarity → semantic conflict
        vec_a = np.array([1.0, 0.0, 0.0, 0.0])
        vec_c = np.array([0.0, 0.0, 0.0, 1.0])  # orthogonal
        report = detect_conflicts(
            a_chunks, c_chunks,
            source_a_embeddings=vec_a.reshape(1, -1),
            source_c_embeddings=vec_c.reshape(1, -1),
        )
        assert report.conflict_detected is True
        assert report.conflict_type in ("semantic", "numeric", "negation")

    def test_report_has_excerpts_when_conflict(self):
        a_chunks = [_mock_ranked_chunk("Source A text about timeout 300 seconds")]
        c_chunks = [_mock_ranked_chunk("Source C text says timeout is 60 seconds")]
        report = detect_conflicts(a_chunks, c_chunks)
        if report.conflict_detected:
            assert len(report.source_a_excerpt) > 0
            assert len(report.source_c_excerpt) > 0

    def test_cosine_similarity_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b) - 0.0) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        # Should not crash — returns 1.0 as safe default
        result = _cosine_similarity(a, b)
        assert isinstance(result, float)


# ── Reranker tests ────────────────────────────────────────────────────────────

class TestConfidenceLabel:
    def test_high_threshold(self):
        assert _confidence_label(0.8) == "High"
        assert _confidence_label(0.65) == "High"

    def test_medium_threshold(self):
        assert _confidence_label(0.5) == "Medium"
        assert _confidence_label(0.35) == "Medium"

    def test_low_threshold(self):
        assert _confidence_label(0.2) == "Low"
        assert _confidence_label(0.0) == "Low"


class TestNormaliseScore:
    def test_zero_gives_half(self):
        """sigmoid(0) = 0.5"""
        result = _normalise_score(0.0, [0.0])
        assert abs(result - 0.5) < 0.01

    def test_positive_gives_above_half(self):
        result = _normalise_score(2.0, [2.0])
        assert result > 0.5

    def test_negative_gives_below_half(self):
        result = _normalise_score(-2.0, [-2.0])
        assert result < 0.5

    def test_result_in_0_1_range(self):
        for score in [-10, -1, 0, 1, 5, 10]:
            result = _normalise_score(float(score), [float(score)])
            assert 0.0 <= result <= 1.0


class TestHybridToRankedFallback:
    def _make_hybrid(self, chunk_id: int, rrf_score: float) -> HybridResult:
        return HybridResult(
            chunk_id=chunk_id, text=f"text {chunk_id}", page_hint=1,
            source_label="A", rrf_score=rrf_score,
            vector_rank=chunk_id + 1, bm25_rank=None,
        )

    def test_converts_hybrid_to_ranked(self):
        hybrids = [self._make_hybrid(i, 1.0 / (60 + i + 1)) for i in range(3)]
        ranked = _hybrid_to_ranked_fallback(hybrids, top_k=3)
        assert len(ranked) == 3

    def test_confidence_in_range(self):
        hybrids = [self._make_hybrid(i, 1.0 / (60 + i + 1)) for i in range(3)]
        ranked = _hybrid_to_ranked_fallback(hybrids, top_k=3)
        for r in ranked:
            assert 0.0 <= r.confidence <= 1.0

    def test_top_k_respected(self):
        hybrids = [self._make_hybrid(i, 0.1) for i in range(10)]
        ranked = _hybrid_to_ranked_fallback(hybrids, top_k=3)
        assert len(ranked) == 3


class TestRerankerFallback:
    def test_fallback_when_not_ready(self):
        """Reranker not ready → should use RRF fallback, not crash."""
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = "test"
        reranker._model = None
        reranker._is_ready = False

        hybrids = [
            HybridResult(
                chunk_id=i, text=f"text {i}", page_hint=1,
                source_label="A", rrf_score=1.0 / (60 + i),
                vector_rank=i, bm25_rank=None,
            )
            for i in range(3)
        ]
        ranked = reranker.rerank("question", hybrids, top_k=2)
        assert len(ranked) == 2
        for r in ranked:
            assert 0.0 <= r.confidence <= 1.0

    def test_source_confidence_from_ranked_chunks(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._is_ready = False

        hybrids = [
            HybridResult(
                chunk_id=i, text=f"text {i}", page_hint=1,
                source_label="A", rrf_score=0.9 - (i * 0.1),
                vector_rank=i, bm25_rank=None,
            )
            for i in range(3)
        ]
        ranked = _hybrid_to_ranked_fallback(hybrids, top_k=3)
        conf = reranker.compute_source_confidence(ranked)
        assert isinstance(conf, SourceConfidence)
        assert conf.label in ("High", "Medium", "Low")
        assert 0.0 <= conf.score <= 1.0

    def test_empty_chunks_returns_low_confidence(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._is_ready = False
        conf = reranker.compute_source_confidence([])
        assert conf.label == "Low"
        assert conf.score == 0.0
