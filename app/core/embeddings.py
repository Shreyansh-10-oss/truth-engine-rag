"""
core/embeddings.py — Embedding Engine Singleton.
Identical to RAG-Lite: load all-MiniLM-L6-v2 once at startup, reuse forever.
"""

import logging
from threading import Lock
from typing import ClassVar

import numpy as np

from app.utils.exceptions import EmbeddingError, EmptyEmbeddingInputError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    _instance: ClassVar["EmbeddingEngine | None"] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dim: int = 384

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> "EmbeddingEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = cls(model_name, batch_size)
                    instance._load_model()
                    cls._instance = instance
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmptyEmbeddingInputError("texts list is empty.")
        cleaned = [t.strip() for t in texts]
        if not any(cleaned):
            raise EmptyEmbeddingInputError("All provided texts are empty strings.")
        safe_texts = [t if t else "[EMPTY]" for t in cleaned]
        try:
            return self._encode_in_batches(safe_texts)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> None:
        logger.info("Loading embedding model '%s'...", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            probe = self._model.encode(["probe"], convert_to_numpy=True)
            self._dim = probe.shape[1]
            logger.info("Embedding model ready. Dim=%d.", self._dim)
        except Exception as exc:
            raise EmbeddingError(f"Could not load model '{self._model_name}': {exc}") from exc

    def _encode_in_batches(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            vecs = self._model.encode(
                batch, convert_to_numpy=True,
                show_progress_bar=False, normalize_embeddings=True,
            )
            all_embeddings.append(vecs)
        return np.vstack(all_embeddings).astype(np.float32)


def get_embedding_engine() -> EmbeddingEngine:
    from config import config
    return EmbeddingEngine.get_instance(
        model_name=config.embedding.model_name,
        batch_size=config.embedding.batch_size,
    )
