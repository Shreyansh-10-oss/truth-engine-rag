"""
api/routes/chat.py — POST /chat (Tier 2)
Now includes confidence scores and conflict information in the response.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.session_store import SessionStore, get_session_store
from app.core.embeddings import EmbeddingEngine, get_embedding_engine
from app.core.llm_router import LLMRouter
from app.core.reranker import CrossEncoderReranker, SourceConfidence, get_reranker
from app.core.truth_engine import TruthEngine, TruthEngineResponse
from app.utils.exceptions import FatalLLMError, SourceNotLoadedError
from config import config

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str
    llm_provider: str = Field(default="gemini")


class SourceChunkOut(BaseModel):
    chunk_id: int
    text: str
    page_hint: int
    source_label: str
    score: float
    confidence: float       # Tier 2: normalised CE score
    confidence_label: str   # Tier 2: High / Medium / Low


class SourceConfidenceOut(BaseModel):
    source_label: str
    score: float
    label: str              # High / Medium / Low
    top_chunk_score: float


class SourceAnswerOut(BaseModel):
    source_label: str
    loaded: bool
    summary: str
    chunks: list[SourceChunkOut]
    confidence: SourceConfidenceOut | None   # Tier 2


class ConflictOut(BaseModel):
    conflict_detected: bool
    severity: str           # 'none', 'low', 'medium', 'high'
    conflict_type: str
    source_a_excerpt: str
    source_c_excerpt: str
    explanation: str
    numeric_conflicts: list[str]


class ChatResponse(BaseModel):
    query: str
    session_id: str
    source_a: SourceAnswerOut | None
    source_b: SourceAnswerOut | None
    source_c: SourceAnswerOut | None
    conclusion: str
    provider_used: str
    fallback_triggered: bool
    conflict: ConflictOut           # Tier 2: full conflict details
    sources_loaded: list[str]


def _get_llm_router() -> LLMRouter:
    from app.core.llm_router import build_router_from_config
    if not hasattr(_get_llm_router, "_instance"):
        _get_llm_router._instance = build_router_from_config()
    return _get_llm_router._instance


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session_store: SessionStore = Depends(get_session_store),
    embedder: EmbeddingEngine = Depends(get_embedding_engine),
    llm_router: LLMRouter = Depends(_get_llm_router),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> ChatResponse:

    session = session_store.get(req.session_id)
    if session is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found.")

    valid_providers = ["gemini", "groq", "ollama"]
    provider = req.llm_provider.lower().strip()
    if provider not in valid_providers:
        raise HTTPException(400, f"Unknown provider '{provider}'.")

    engine = TruthEngine(
        embedding_engine=embedder,
        faiss_stores=session.get_faiss_dict(),
        bm25_stores=session.get_bm25_dict(),
        llm_router=llm_router,
        reranker=reranker,
        top_k=config.faiss.top_k,
        hybrid_candidates=config.faiss.top_k * 3,
    )

    try:
        result: TruthEngineResponse = engine.query(
            user_question=req.query,
            preferred_provider=provider,
        )
    except SourceNotLoadedError as exc:
        raise HTTPException(409, exc.message)
    except FatalLLMError as exc:
        raise HTTPException(503, exc.message)
    except Exception as exc:
        logger.error("Truth Engine error: %s", exc)
        raise HTTPException(500, f"Internal error: {exc}")

    def _build_source_out(sr) -> SourceAnswerOut | None:
        if sr is None:
            return None
        return SourceAnswerOut(
            source_label=sr.source_label,
            loaded=sr.loaded,
            summary=sr.summary,
            chunks=[
                SourceChunkOut(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    page_hint=c.page_hint,
                    source_label=c.source_label,
                    score=c.ce_score,
                    confidence=c.confidence,
                    confidence_label=c.confidence_label,
                )
                for c in sr.chunks
            ],
            confidence=SourceConfidenceOut(
                source_label=sr.confidence.source_label,
                score=sr.confidence.score,
                label=sr.confidence.label,
                top_chunk_score=sr.confidence.top_chunk_score,
            ) if sr.confidence else None,
        )

    # Build conflict output
    cr = result.conflict_report
    conflict_out = ConflictOut(
        conflict_detected=result.conflict_detected,
        severity=cr.severity if cr else "none",
        conflict_type=cr.conflict_type if cr else "none",
        source_a_excerpt=cr.source_a_excerpt if cr else "",
        source_c_excerpt=cr.source_c_excerpt if cr else "",
        explanation=cr.explanation if cr else "",
        numeric_conflicts=cr.numeric_conflicts if cr else [],
    )

    return ChatResponse(
        query=req.query,
        session_id=req.session_id,
        source_a=_build_source_out(result.source_a),
        source_b=_build_source_out(result.source_b),
        source_c=_build_source_out(result.source_c),
        conclusion=result.conclusion,
        provider_used=result.provider_used,
        fallback_triggered=result.fallback_triggered,
        conflict=conflict_out,
        sources_loaded=session.sources_loaded(),
    )
