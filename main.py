"""
main.py — FastAPI entry point for Truth Engine Tier 2.
Warms embedding model AND cross-encoder reranker at startup.
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, ingest
from app.api.session_store import get_session_store
from app.core.embeddings import get_embedding_engine
from app.core.llm_router import LLMRouter, build_router_from_config
from app.core.reranker import get_reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Truth Engine Tier 2 startup ===")

    from config import config
    logger.info("Config loaded. Debug=%s", config.debug)

    logger.info("Warming embedding engine (all-MiniLM-L6-v2)...")
    embedder = get_embedding_engine()
    logger.info("Embedding engine ready. Dim=%d.", embedder.dim)

    logger.info("Warming cross-encoder reranker...")
    reranker = get_reranker()
    if reranker.is_ready:
        logger.info("Cross-encoder reranker ready.")
    else:
        logger.warning("Cross-encoder not loaded — falling back to RRF scores.")

    store = get_session_store()
    logger.info("Session store ready (max=%d sessions).", store.capacity)

    router: LLMRouter = build_router_from_config()
    if config.debug:
        health = router.health_status()
        for provider, ok in health.items():
            logger.info("  Provider '%s': %s", provider, "✓" if ok else "✗")

    app.state.llm_router = router
    logger.info("=== Startup complete. Listening... ===")
    yield
    logger.info("=== Truth Engine Tier 2 shutdown ===")


app = FastAPI(
    title="Truth Engine API — Tier 2",
    version="2.0.0",
    description=(
        "Multi-source RAG with Hybrid Search (BM25+FAISS), "
        "Cross-Encoder Re-Ranking, Confidence Scoring, and Conflict Detection."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(chat.router,   tags=["Chat"])


@app.get("/health", tags=["System"])
async def health():
    router: LLMRouter = build_router_from_config()
    reranker = get_reranker()
    return {
        "status": "ok",
        "tier": 2,
        "providers_available": router.health_status(),
        "reranker_ready": reranker.is_ready,
    }


@app.get("/sessions", tags=["System"])
async def list_sessions():
    store = get_session_store()
    return {"sessions": store.list_all(), "count": store.count}


@app.delete("/session/{session_id}", tags=["System"])
async def delete_session(session_id: str):
    store = get_session_store()
    deleted = store.delete(session_id)
    if not deleted:
        from fastapi import HTTPException
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
