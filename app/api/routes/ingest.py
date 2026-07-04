"""
api/routes/ingest.py — POST /ingest/{source_label}
Tier 2: builds BOTH FAISSStore (vector) and BM25Store (keyword) during ingest.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from pydantic import BaseModel

from app.api.session_store import SessionData, SessionStore, SourceInfo, get_session_store
from app.core.embeddings import EmbeddingEngine, get_embedding_engine
from app.db.bm25_store import BM25Store
from app.db.faiss_store import FAISSStore
from app.utils.exceptions import (
    ChunkingError, EmbeddingError, FileParseError, IndexBuildError, UnsupportedFileTypeError,
)
from app.utils.file_parser import SUPPORTED_EXTENSIONS, detect_file_type, parse_file_by_page
from app.utils.text_splitter import split_text
from config import config

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
VALID_SOURCES = {"A", "B", "C"}


class IngestResponse(BaseModel):
    status: str
    session_id: str
    source_label: str
    filename: str
    file_type: str
    chunk_count: int
    sources_loaded: list[str]


@router.post("/ingest/{source_label}", response_model=IngestResponse)
async def ingest_source(
    source_label: str = Path(...),
    file: UploadFile = File(...),
    session_id: str | None = None,
    session_store: SessionStore = Depends(get_session_store),
    embedder: EmbeddingEngine = Depends(get_embedding_engine),
) -> IngestResponse:
    source_label = source_label.upper().strip()
    if source_label not in VALID_SOURCES:
        raise HTTPException(400, f"Invalid source '{source_label}'. Must be A, B, or C.")

    filename = file.filename or "upload"
    ext = detect_file_type(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File exceeds 20 MB limit.")

    if not session_id:
        session_id = str(uuid.uuid4())
    session: SessionData = session_store.get_or_create(session_id)

    try:
        page_texts = parse_file_by_page(file_bytes, filename)
        full_text  = "\n\n".join(p for p in page_texts if p)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(400, exc.message)
    except FileParseError as exc:
        raise HTTPException(422, exc.message)

    try:
        chunks = split_text(
            text=full_text, source_label=source_label, page_texts=page_texts,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            min_chunk_tokens=config.chunking.min_chunk_tokens,
        )
    except ChunkingError as exc:
        raise HTTPException(422, exc.message)

    try:
        embeddings = embedder.embed([c["text"] for c in chunks])
    except EmbeddingError as exc:
        raise HTTPException(500, f"Embedding failed: {exc}")

    # FAISS (vector search)
    try:
        faiss_store = FAISSStore(embedding_dim=embedder.dim, source_label=source_label)
        faiss_store.build(embeddings, chunks)
    except IndexBuildError as exc:
        raise HTTPException(500, f"FAISS build failed: {exc}")

    # BM25 (keyword search) — non-fatal if fails
    bm25_store = BM25Store(source_label=source_label)
    try:
        bm25_store.build(chunks)
        logger.info("[Source %s] BM25 index built alongside FAISS.", source_label)
    except Exception as exc:
        logger.warning("[Source %s] BM25 build failed (non-fatal): %s", source_label, exc)

    info = SourceInfo(filename=filename, file_type=ext.lstrip("."), chunk_count=len(chunks))
    session.set_stores(source_label, faiss_store, bm25_store, info)

    logger.info("[Source %s] Ingestion complete: session='%s' chunks=%d.", source_label, session_id, len(chunks))

    return IngestResponse(
        status="ok", session_id=session_id, source_label=source_label,
        filename=filename, file_type=ext.lstrip("."),
        chunk_count=len(chunks), sources_loaded=session.sources_loaded(),
    )
