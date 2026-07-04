"""
db/faiss_store.py — FAISS Vector Store.
Identical to Tier 1. SearchResult includes source_label.
"""

import logging
from typing import TypedDict

import faiss
import numpy as np

from app.utils.exceptions import IndexBuildError, IndexNotReadyError
from app.utils.text_splitter import ChunkDict

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    chunk_id: int
    text: str
    page_hint: int
    score: float
    source_label: str


class FAISSStore:
    def __init__(self, embedding_dim: int = 384, source_label: str = "A") -> None:
        self._dim = embedding_dim
        self._source_label = source_label
        self._index: faiss.IndexFlatL2 | None = None
        self._metadata: dict[int, ChunkDict] = {}
        self._is_ready = False

    def build(self, embeddings: np.ndarray, chunks: list[ChunkDict]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise IndexBuildError(
                f"[Source {self._source_label}] Embedding count ({embeddings.shape[0]}) "
                f"!= chunk count ({len(chunks)})."
            )
        if embeddings.shape[1] != self._dim:
            raise IndexBuildError(
                f"[Source {self._source_label}] Dim mismatch: expected {self._dim}, got {embeddings.shape[1]}."
            )
        try:
            index = faiss.IndexFlatL2(self._dim)
            index.add(embeddings.astype(np.float32))
        except Exception as exc:
            raise IndexBuildError(f"[Source {self._source_label}] FAISS build failed: {exc}") from exc

        self._metadata = {i: chunk for i, chunk in enumerate(chunks)}
        self._index = index
        self._is_ready = True
        logger.info("[Source %s] FAISSStore ready: %d vectors.", self._source_label, index.ntotal)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if not self._is_ready or self._index is None:
            raise IndexNotReadyError(
                f"Source {self._source_label} index not ready."
            )
        vec = query_vector.reshape(1, -1).astype(np.float32)
        k_clamped = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k_clamped)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[int(idx)]
            results.append(SearchResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                page_hint=chunk["page_hint"],
                score=float(dist),
                source_label=chunk.get("source_label", self._source_label),
            ))
        return results

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def chunk_count(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def source_label(self) -> str:
        return self._source_label

    def reset(self) -> None:
        self._index = None
        self._metadata = {}
        self._is_ready = False
        logger.info("[Source %s] FAISSStore reset.", self._source_label)
