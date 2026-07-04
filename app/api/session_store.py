"""
api/session_store.py — Multi-source session registry with LRU eviction.
Tier 2 upgrade: each source now holds BOTH a FAISSStore and a BM25Store.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from app.db.bm25_store import BM25Store
from app.db.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


@dataclass
class SourceInfo:
    filename: str
    file_type: str
    chunk_count: int
    loaded: bool = True


@dataclass
class SourceStores:
    """Holds both index types for a single source."""
    faiss_store: FAISSStore
    bm25_store: BM25Store
    info: SourceInfo


@dataclass
class SessionData:
    session_id: str
    stores_a: SourceStores | None = None
    stores_b: SourceStores | None = None
    stores_c: SourceStores | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    def get_stores(self, label: str) -> SourceStores | None:
        return {"A": self.stores_a, "B": self.stores_b, "C": self.stores_c}.get(label)

    def set_stores(self, label: str, faiss: FAISSStore, bm25: BM25Store, info: SourceInfo) -> None:
        ss = SourceStores(faiss_store=faiss, bm25_store=bm25, info=info)
        if label == "A":
            self.stores_a = ss
        elif label == "B":
            self.stores_b = ss
        elif label == "C":
            self.stores_c = ss

    def get_faiss_dict(self) -> dict[str, FAISSStore]:
        result = {}
        for label, ss in [("A", self.stores_a), ("B", self.stores_b), ("C", self.stores_c)]:
            if ss is not None:
                result[label] = ss.faiss_store
        return result

    def get_bm25_dict(self) -> dict[str, BM25Store]:
        result = {}
        for label, ss in [("A", self.stores_a), ("B", self.stores_b), ("C", self.stores_c)]:
            if ss is not None:
                result[label] = ss.bm25_store
        return result

    def sources_loaded(self) -> list[str]:
        loaded = []
        for label, ss in [("A", self.stores_a), ("B", self.stores_b), ("C", self.stores_c)]:
            if ss is not None and ss.faiss_store.is_ready:
                loaded.append(label)
        return loaded

    def total_chunks(self) -> int:
        total = 0
        for ss in [self.stores_a, self.stores_b, self.stores_c]:
            if ss:
                total += ss.faiss_store.chunk_count
        return total


class SessionStore:
    def __init__(self, max_sessions: int = 3) -> None:
        self._max = max_sessions
        self._store: OrderedDict[str, SessionData] = OrderedDict()
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id in self._store:
                data = self._store[session_id]
                data.last_accessed = datetime.utcnow()
                self._store.move_to_end(session_id)
                return data
            if len(self._store) >= self._max:
                evicted_id, _ = self._store.popitem(last=False)
                logger.info("SessionStore: evicted session '%s'.", evicted_id)
            data = SessionData(session_id=session_id)
            self._store[session_id] = data
            self._store.move_to_end(session_id)
            return data

    def get(self, session_id: str) -> SessionData | None:
        with self._lock:
            data = self._store.get(session_id)
            if data is not None:
                data.last_accessed = datetime.utcnow()
                self._store.move_to_end(session_id)
            return data

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def list_all(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "sources_loaded": s.sources_loaded(),
                    "total_chunks": s.total_chunks(),
                    "created_at": s.created_at.isoformat(),
                    "last_accessed": s.last_accessed.isoformat(),
                }
                for s in self._store.values()
            ]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def capacity(self) -> int:
        return self._max


_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        from config import config
        _session_store = SessionStore(max_sessions=config.session.max_sessions)
    return _session_store
