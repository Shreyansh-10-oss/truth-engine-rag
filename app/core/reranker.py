"""
core/reranker.py — Cross-Encoder Re-Ranker (Tier 2)
Responsibility: Re-score retrieved chunks using a cross-encoder model that
reads both the query AND the chunk together — far more accurate than
embedding similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80 MB, CPU-friendly, runs in ~10ms per chunk
  - Trained on MS MARCO passage ranking — excellent for Q&A tasks
  - Returns a relevance logit; we normalise to a 0-1 confidence score

Pattern: Singleton (loaded once at startup, same as EmbeddingEngine).

Also computes per-source ConfidenceScore used in the UI.
"""

import logging
import math
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar

from app.core.hybrid_retriever import HybridResult

logger = logging.getLogger(__name__)

# Confidence thresholds (based on normalised cross-encoder score)
CONFIDENCE_HIGH   = 0.65
CONFIDENCE_MEDIUM = 0.35


@dataclass
class RankedChunk:
    """A chunk after cross-encoder re-ranking."""
    chunk_id: int
    text: str
    page_hint: int
    source_label: str
    ce_score: float       # Raw cross-encoder logit
    confidence: float     # Normalised 0.0–1.0
    confidence_label: str # 'High', 'Medium', or 'Low'
    rrf_score: float      # Original hybrid score (preserved for reference)


@dataclass
class SourceConfidence:
    """Aggregate confidence for an entire source's answer."""
    source_label: str
    score: float          # Average confidence of top chunks
    label: str            # 'High', 'Medium', or 'Low'
    top_chunk_score: float


class CrossEncoderReranker:
    """
    Singleton cross-encoder re-ranker.
    Load once at startup, reuse for every query.
    """

    _instance: ClassVar["CrossEncoderReranker | None"] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._is_ready = False

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> "CrossEncoderReranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls(model_name)
                    inst._load_model()
                    cls._instance = inst
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        chunks: list[HybridResult],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """
        Re-score chunks using the cross-encoder and return top_k sorted by score.

        Args:
            query:  The user's question.
            chunks: Candidate chunks from hybrid retrieval.
            top_k:  How many to return after re-ranking.

        Returns:
            List of RankedChunk sorted by descending confidence.
        """
        if not chunks:
            return []

        if not self._is_ready or self._model is None:
            logger.warning("Reranker not ready — returning hybrid results as-is.")
            return _hybrid_to_ranked_fallback(chunks, top_k)

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error("Cross-encoder prediction failed: %s — using fallback.", exc)
            return _hybrid_to_ranked_fallback(chunks, top_k)

        # Pair scores with chunks
        scored = list(zip(raw_scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Normalise scores and build RankedChunk list
        all_scores = [s for s, _ in scored]
        ranked: list[RankedChunk] = []
        for raw_score, chunk in scored[:top_k]:
            conf  = _normalise_score(float(raw_score), all_scores)
            label = _confidence_label(conf)
            ranked.append(RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_hint=chunk.page_hint,
                source_label=chunk.source_label,
                ce_score=float(raw_score),
                confidence=conf,
                confidence_label=label,
                rrf_score=chunk.rrf_score,
            ))

        logger.debug(
            "Reranker: %d chunks → top %d. Best score: %.3f (conf: %s).",
            len(chunks), len(ranked),
            ranked[0].ce_score if ranked else 0,
            ranked[0].confidence_label if ranked else "N/A",
        )
        return ranked

    def compute_source_confidence(self, ranked_chunks: list[RankedChunk]) -> SourceConfidence:
        """
        Compute aggregate confidence for a source based on its top re-ranked chunks.
        Uses the average of the top-3 chunk confidence scores.
        """
        if not ranked_chunks:
            return SourceConfidence(
                source_label="?", score=0.0, label="Low", top_chunk_score=0.0
            )

        source_label = ranked_chunks[0].source_label
        top3 = ranked_chunks[:3]
        avg_score = sum(c.confidence for c in top3) / len(top3)
        top_score = ranked_chunks[0].confidence

        return SourceConfidence(
            source_label=source_label,
            score=round(avg_score, 3),
            label=_confidence_label(avg_score),
            top_chunk_score=round(top_score, 3),
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        logger.info("Loading cross-encoder model '%s'...", self._model_name)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, max_length=512)
            self._is_ready = True
            logger.info("Cross-encoder model loaded.")
        except Exception as exc:
            logger.error("Failed to load cross-encoder: %s — reranker disabled.", exc)
            self._is_ready = False


# ── Module-level convenience ──────────────────────────────────────────────────

def get_reranker() -> CrossEncoderReranker:
    """FastAPI dependency — returns singleton reranker."""
    return CrossEncoderReranker.get_instance()


# ── Private helpers ───────────────────────────────────────────────────────────

def _normalise_score(score: float, all_scores: list[float]) -> float:
    """
    Normalise a cross-encoder logit to [0, 1] using sigmoid.
    Sigmoid is appropriate because cross-encoders output unbounded logits.
    """
    return round(1.0 / (1.0 + math.exp(-score)), 4)


def _confidence_label(score: float) -> str:
    """Map normalised confidence score to a human-readable label."""
    if score >= CONFIDENCE_HIGH:
        return "High"
    elif score >= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


def _hybrid_to_ranked_fallback(
    chunks: list[HybridResult],
    top_k: int,
) -> list[RankedChunk]:
    """
    Fallback when reranker is unavailable.
    Converts HybridResult → RankedChunk using RRF score as proxy confidence.
    """
    ranked = []
    max_rrf = chunks[0].rrf_score if chunks else 1.0
    for chunk in chunks[:top_k]:
        conf  = min(chunk.rrf_score / max(max_rrf, 1e-9), 1.0)
        ranked.append(RankedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            page_hint=chunk.page_hint,
            source_label=chunk.source_label,
            ce_score=chunk.rrf_score,
            confidence=round(conf, 4),
            confidence_label=_confidence_label(conf),
            rrf_score=chunk.rrf_score,
        ))
    return ranked
