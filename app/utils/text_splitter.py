"""
utils/text_splitter.py — Text Chunking
Identical strategy to RAG-Lite: fixed-size overlapping chunks, word-boundary splits.
ChunkDict now includes 'source_label' (A/B/C) for the Truth Engine.
"""

import logging
import re
from typing import TypedDict

from app.utils.exceptions import ChunkingError

logger = logging.getLogger(__name__)


class ChunkDict(TypedDict):
    chunk_id: int
    text: str
    page_hint: int       # 1-based approximate page/section number
    source_label: str    # 'A', 'B', or 'C'


def split_text(
    text: str,
    source_label: str = "A",
    page_texts: list[str] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_tokens: int = 20,
) -> list[ChunkDict]:
    """
    Split raw text into overlapping fixed-size chunks with source metadata.

    Args:
        text:          Full document text.
        source_label:  'A', 'B', or 'C' — stamped onto every chunk.
        page_texts:    Optional per-page/section strings for page_hint.
        chunk_size:    Target chunk size in approximate tokens.
        chunk_overlap: Overlap between chunks in approximate tokens.
        min_chunk_tokens: Minimum chunk size; shorter ones are merged.

    Returns:
        List of ChunkDict, each with chunk_id, text, page_hint, source_label.

    Raises:
        ChunkingError: If no valid chunks can be produced.
    """
    if not text or not text.strip():
        raise ChunkingError(f"[Source {source_label}] Input text is empty.")

    char_size    = chunk_size    * 4
    char_overlap = chunk_overlap * 4
    min_chars    = min_chunk_tokens * 4

    raw_chunks = _sliding_window(text, char_size, char_overlap)
    merged     = _merge_short_chunks(raw_chunks, min_chars)

    if not merged:
        raise ChunkingError(f"[Source {source_label}] Chunking produced no valid chunks.")

    page_boundary_map = _build_page_boundary_map(page_texts) if page_texts else {}

    chunks: list[ChunkDict] = []
    for idx, chunk_text in enumerate(merged):
        chunks.append(ChunkDict(
            chunk_id=idx,
            text=chunk_text,
            page_hint=_estimate_page(chunk_text, text, page_boundary_map),
            source_label=source_label,
        ))

    logger.info(
        "[Source %s] %d chunks produced (size=%d, overlap=%d).",
        source_label, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


# ── Private helpers (identical to RAG-Lite) ───────────────────────────────────

def _sliding_window(text: str, char_size: int, char_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + char_size, text_len)
        if end < text_len:
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        step = char_size - char_overlap
        next_start = start + step
        if next_start < text_len:
            snap = text.rfind(" ", start, next_start + 1)
            if snap != -1 and snap > start:
                next_start = snap + 1
        start = next_start
    return chunks


def _merge_short_chunks(chunks: list[str], min_chars: int) -> list[str]:
    if not chunks:
        return chunks
    merged: list[str] = []
    for chunk in chunks:
        if len(chunk) < min_chars and merged:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def _build_page_boundary_map(page_texts: list[str]) -> dict[str, int]:
    boundary_map: dict[str, int] = {}
    for page_num, page_text in enumerate(page_texts, start=1):
        if page_text:
            key = page_text[:60].strip()
            if key:
                boundary_map[key] = page_num
    return boundary_map


def _estimate_page(chunk_text: str, full_text: str, page_boundary_map: dict[str, int]) -> int:
    if not page_boundary_map:
        return 1
    pos = full_text.find(chunk_text[:50])
    if pos == -1:
        return 1
    cumulative = 0
    for page_num, page_text in enumerate(
        sorted(page_boundary_map.items(), key=lambda x: x[1]), start=1
    ):
        cumulative += len(page_text[0])
        if cumulative >= pos:
            return page_num
    return max(page_boundary_map.values()) if page_boundary_map else 1
