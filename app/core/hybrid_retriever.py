"""
core/hybrid_retriever.py — Hybrid Search via Reciprocal Rank Fusion (Tier 2)
Responsibility: Combine BM25 keyword results + FAISS vector results into a
single ranked list per source using RRF.

Why RRF?
  - No extra model or weights to tune
  - Proven to outperform both individual methods in IR benchmarks
  - Handles different score scales (BM25 vs L2 distance) naturally
  - Formula: RRF(d) = Σ 1 / (k + rank(d))  where k=60 is standard

Architecture:
  One HybridRetriever is created per source per query — it is stateless
  and takes pre-built FAISSStore + BM25Store as inputs.
"""

import logging
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from app.db.bm25_store import BM25Store
from app.db.faiss_store import FAISSStore, SearchResult

logger = logging.getLogger(__name__)

# RRF constant — 60 is the standard value from the original RRF paper
RRF_K = 60


@dataclass
class HybridResult:
    """A single chunk retrieved by hybrid search with fused rank score."""
    chunk_id: int
    text: str
    page_hint: int
    source_label: str
    rrf_score: float        # Higher = more relevant (fused rank score)
    vector_rank: int | None  # Rank in FAISS results (1-based), None if not retrieved
    bm25_rank: int | None    # Rank in BM25 results (1-based), None if not retrieved


class HybridRetriever:
    """
    Combines FAISS vector search + BM25 keyword search using RRF.

    Usage:
        retriever = HybridRetriever(faiss_store, bm25_store)
        results = retriever.retrieve(query_vector, query_text, k=5)
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_store: BM25Store,
        vector_k: int = 10,    # Fetch more candidates before fusion
        bm25_k: int = 10,
    ) -> None:
        self._faiss = faiss_store
        self._bm25  = bm25_store
        self._vector_k = vector_k
        self._bm25_k   = bm25_k

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
    ) -> list[HybridResult]:
        """
        Run hybrid search and return top_k fused results.

        Args:
            query_vector: Embedded query (from EmbeddingEngine).
            query_text:   Raw query string (for BM25).
            top_k:        Number of final results to return.

        Returns:
            List of HybridResult sorted by descending RRF score.
        """
        source = self._faiss.source_label

        # ── Step 1: Run both searches ─────────────────────────────────────────
        vector_results = []
        if self._faiss.is_ready:
            try:
                vector_results = self._faiss.search(query_vector, k=self._vector_k)
            except Exception as exc:
                logger.warning("[Source %s] FAISS search failed: %s", source, exc)

        bm25_results = []
        if self._bm25.is_ready:
            try:
                bm25_results = self._bm25.search(query_text, k=self._bm25_k)
            except Exception as exc:
                logger.warning("[Source %s] BM25 search failed: %s", source, exc)

        # If only one method available, return its results directly
        if not vector_results and not bm25_results:
            return []

        if not vector_results:
            return _bm25_to_hybrid(bm25_results, top_k)

        if not bm25_results:
            return _faiss_to_hybrid(vector_results, top_k)

        # ── Step 2: Build chunk_id → rank maps ────────────────────────────────
        vector_rank_map: dict[int, int] = {
            r["chunk_id"]: rank + 1
            for rank, r in enumerate(vector_results)
        }
        bm25_rank_map: dict[int, int] = {
            r["chunk_id"]: rank + 1
            for rank, r in enumerate(bm25_results)
        }

        # ── Step 3: Collect all unique chunk IDs ──────────────────────────────
        all_chunk_ids = set(vector_rank_map.keys()) | set(bm25_rank_map.keys())

        # ── Step 4: Build lookup for chunk metadata ───────────────────────────
        chunk_lookup: dict[int, dict] = {}
        for r in vector_results:
            chunk_lookup[r["chunk_id"]] = {
                "text": r["text"],
                "page_hint": r["page_hint"],
                "source_label": r["source_label"],
            }
        for r in bm25_results:
            if r["chunk_id"] not in chunk_lookup:
                chunk_lookup[r["chunk_id"]] = {
                    "text": r["text"],
                    "page_hint": r["page_hint"],
                    "source_label": r["source_label"],
                }

        # ── Step 5: Compute RRF score for every candidate ─────────────────────
        fused: list[HybridResult] = []
        for cid in all_chunk_ids:
            v_rank = vector_rank_map.get(cid)
            b_rank = bm25_rank_map.get(cid)

            rrf = 0.0
            if v_rank is not None:
                rrf += 1.0 / (RRF_K + v_rank)
            if b_rank is not None:
                rrf += 1.0 / (RRF_K + b_rank)

            meta = chunk_lookup.get(cid, {})
            fused.append(HybridResult(
                chunk_id=cid,
                text=meta.get("text", ""),
                page_hint=meta.get("page_hint", 1),
                source_label=meta.get("source_label", source),
                rrf_score=rrf,
                vector_rank=v_rank,
                bm25_rank=b_rank,
            ))

        # ── Step 6: Sort by RRF score descending, return top_k ────────────────
        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        top = fused[:top_k]

        logger.info(
            "[Source %s] Hybrid retrieval: %d vector + %d BM25 → %d fused (top %d).",
            source, len(vector_results), len(bm25_results), len(fused), len(top),
        )
        return top


# ── Fallback converters ───────────────────────────────────────────────────────

def _faiss_to_hybrid(results: list[SearchResult], top_k: int) -> list[HybridResult]:
    """Convert FAISS-only results to HybridResult when BM25 unavailable."""
    out = []
    for rank, r in enumerate(results[:top_k], start=1):
        out.append(HybridResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            page_hint=r["page_hint"],
            source_label=r["source_label"],
            rrf_score=1.0 / (RRF_K + rank),
            vector_rank=rank,
            bm25_rank=None,
        ))
    return out


def _bm25_to_hybrid(results, top_k: int) -> list[HybridResult]:
    """Convert BM25-only results to HybridResult when FAISS unavailable."""
    out = []
    for rank, r in enumerate(results[:top_k], start=1):
        out.append(HybridResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            page_hint=r["page_hint"],
            source_label=r["source_label"],
            rrf_score=1.0 / (RRF_K + rank),
            vector_rank=None,
            bm25_rank=rank,
        ))
    return out
