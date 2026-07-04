"""
db/bm25_store.py — BM25 Keyword Index (Tier 2)
Responsibility: Fast keyword-based retrieval to complement FAISS vector search.
Uses rank_bm25 library — no model required, pure token matching.

One BM25Store per source (A, B, C), built in parallel with FAISSStore during ingest.
Used by HybridRetriever to combine keyword + semantic results via RRF fusion.
"""

import logging
import re
import string
from typing import TypedDict

from app.utils.text_splitter import ChunkDict

logger = logging.getLogger(__name__)


class BM25Result(TypedDict):
    chunk_id: int
    text: str
    page_hint: int
    source_label: str
    bm25_score: float


class BM25Store:
    """
    In-memory BM25 keyword index for a single source.

    Lifecycle:
        store = BM25Store(source_label='A')
        store.build(chunks)
        results = store.search("timeout seconds", k=5)
    """

    def __init__(self, source_label: str = "A") -> None:
        self._source_label = source_label
        self._bm25 = None
        self._chunks: list[ChunkDict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._is_ready = False

    def build(self, chunks: list[ChunkDict]) -> None:
        """
        Build the BM25 index from a list of chunks.

        Args:
            chunks: Same ChunkDict list used to build the FAISSStore.
        """
        if not chunks:
            logger.warning("[Source %s] BM25Store: no chunks to index.", self._source_label)
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for Tier 2 hybrid search. "
                "Run: pip install rank-bm25"
            )

        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._is_ready = True

        logger.info(
            "[Source %s] BM25Store ready: %d documents indexed.",
            self._source_label, len(chunks),
        )

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        """
        Retrieve top-k chunks by BM25 keyword score.

        Args:
            query: Raw query string (tokenized internally).
            k:     Number of results to return.

        Returns:
            List of BM25Result sorted by descending BM25 score.
        """
        if not self._is_ready or self._bm25 is None:
            logger.warning("[Source %s] BM25Store not ready — skipping keyword search.", self._source_label)
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with chunk indices, sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_k  = scored[:k]

        results: list[BM25Result] = []
        for idx, score in top_k:
            if score <= 0:
                continue   # Skip zero-score results — no keyword overlap
            chunk = self._chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                page_hint=chunk["page_hint"],
                source_label=chunk.get("source_label", self._source_label),
                bm25_score=float(score),
            ))

        logger.debug(
            "[Source %s] BM25 search: %d results for query '%s'.",
            self._source_label, len(results), query[:50],
        )
        return results

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def source_label(self) -> str:
        return self._source_label

    def reset(self) -> None:
        self._bm25 = None
        self._chunks = []
        self._tokenized_corpus = []
        self._is_ready = False
        logger.info("[Source %s] BM25Store reset.", self._source_label)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Common English stop words — removing them improves BM25 precision
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "we", "you", "he", "she", "they", "i", "my", "your", "our", "their",
    "not", "no", "as", "if", "so", "than", "then", "when", "where", "which",
    "who", "what", "how", "all", "each", "more", "also", "can", "into",
}


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, remove punctuation, split on whitespace, remove stop words.
    Keeps technical terms like 'ota.update_timeout_seconds' intact.
    """
    text = text.lower()
    # Replace punctuation except dots (preserve dotted tech terms like ota.timeout)
    text = re.sub(r"[^\w\s.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]
