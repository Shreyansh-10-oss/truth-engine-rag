"""
utils/file_parser.py — Multi-format file ingestion.
Accepts PDF, JSON, CSV, Markdown, and TXT — returns plain text.
Isolated: swap any parser without touching other layers.

Source label is attached to every parsed text so the RAG engine
can always trace which source a chunk came from.
"""

import csv
import io
import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.utils.exceptions import (
    JSONParseError,
    MarkdownParseError,
    PDFEncryptedError,
    PDFNoTextError,
    PDFParseError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

# Supported extensions
SUPPORTED_EXTENSIONS = {".pdf", ".json", ".csv", ".md", ".txt"}


def detect_file_type(filename: str) -> str:
    """Return the normalised lowercase extension, e.g. '.pdf'."""
    return Path(filename).suffix.lower()


def parse_file(file_bytes: bytes, filename: str) -> str:
    """
    Parse any supported file type and return a single plain-text string.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename — used to detect type and for logging.

    Returns:
        Extracted plain text ready for chunking.

    Raises:
        UnsupportedFileTypeError: Extension not in SUPPORTED_EXTENSIONS.
        PDFEncryptedError / PDFNoTextError / PDFParseError: PDF-specific errors.
        JSONParseError:     JSON or CSV parsing failed.
        MarkdownParseError: Markdown/TXT parsing failed.
    """
    ext = detect_file_type(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(ext)

    if ext == ".pdf":
        return _parse_pdf(file_bytes, filename)
    elif ext in (".json",):
        return _parse_json(file_bytes, filename)
    elif ext == ".csv":
        return _parse_csv(file_bytes, filename)
    elif ext in (".md", ".txt"):
        return _parse_text(file_bytes, filename)

    raise UnsupportedFileTypeError(ext)


def parse_file_by_page(file_bytes: bytes, filename: str) -> list[str]:
    """
    Same as parse_file but returns a list of page/section-level strings.
    Used for page_hint metadata in chunking.
    For non-PDF formats, each logical section is treated as one 'page'.
    """
    ext = detect_file_type(filename)

    if ext == ".pdf":
        return _parse_pdf_by_page(file_bytes, filename)
    elif ext == ".json":
        return _parse_json_sections(file_bytes, filename)
    elif ext == ".csv":
        return _parse_csv_rows(file_bytes, filename)
    elif ext in (".md", ".txt"):
        return _parse_text_sections(file_bytes, filename)

    raise UnsupportedFileTypeError(ext)


# ── PDF ───────────────────────────────────────────────────────────────────────

def _parse_pdf(file_bytes: bytes, filename: str) -> str:
    pages = _parse_pdf_by_page(file_bytes, filename)
    return "\n\n".join(p for p in pages if p)


def _parse_pdf_by_page(file_bytes: bytes, filename: str) -> list[str]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"Could not open '{filename}': {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise PDFEncryptedError()

    pages: list[str] = []
    for page_num in range(len(doc)):
        try:
            text = doc.load_page(page_num).get_text("text").strip()
            pages.append(_clean_text(text) if text else "")
        except Exception as exc:
            logger.warning("Could not read page %d of '%s': %s", page_num + 1, filename, exc)
            pages.append("")

    doc.close()

    if not any(pages):
        raise PDFNoTextError()

    logger.info("'%s': extracted %d pages from PDF.", filename, len(pages))
    return pages


# ── JSON ──────────────────────────────────────────────────────────────────────

def _parse_json(file_bytes: bytes, filename: str) -> str:
    sections = _parse_json_sections(file_bytes, filename)
    return "\n\n".join(sections)


def _parse_json_sections(file_bytes: bytes, filename: str) -> list[str]:
    """
    Convert JSON to readable text.
    Supports: list of dicts (records), dict, or any JSON structure.
    Each top-level record becomes one 'section'.
    """
    try:
        raw = file_bytes.decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception as exc:
        raise JSONParseError(f"Failed to parse JSON '{filename}': {exc}") from exc

    sections: list[str] = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            sections.append(_dict_to_text(item, record_num=i + 1))
    elif isinstance(data, dict):
        # Treat each top-level key as a section
        for key, val in data.items():
            if isinstance(val, list):
                for i, item in enumerate(val):
                    sections.append(f"[{key} record {i+1}]\n{_dict_to_text(item)}")
            else:
                sections.append(f"[{key}]\n{_dict_to_text(val)}")
    else:
        sections.append(str(data))

    logger.info("'%s': extracted %d sections from JSON.", filename, len(sections))
    return sections if sections else ["(empty JSON file)"]


def _dict_to_text(obj, record_num: int | None = None) -> str:
    """Recursively flatten a dict/list/value to readable key: value lines."""
    prefix = f"[Record {record_num}]\n" if record_num else ""
    if isinstance(obj, dict):
        lines = [f"{k}: {_flatten_value(v)}" for k, v in obj.items()]
        return prefix + "\n".join(lines)
    elif isinstance(obj, list):
        return prefix + "\n".join(str(item) for item in obj)
    else:
        return prefix + str(obj)


def _flatten_value(val) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


# ── CSV ───────────────────────────────────────────────────────────────────────

def _parse_csv(file_bytes: bytes, filename: str) -> str:
    rows = _parse_csv_rows(file_bytes, filename)
    return "\n\n".join(rows)


def _parse_csv_rows(file_bytes: bytes, filename: str) -> list[str]:
    """
    Convert CSV to text. Each row becomes one section:
    'column1: value1 | column2: value2 | ...'
    """
    try:
        raw = file_bytes.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
    except Exception as exc:
        raise JSONParseError(f"Failed to parse CSV '{filename}': {exc}") from exc

    if not rows:
        return ["(empty CSV file)"]

    sections: list[str] = []
    for i, row in enumerate(rows, start=1):
        parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
        sections.append(f"[Row {i}] " + " | ".join(parts))

    logger.info("'%s': extracted %d rows from CSV.", filename, len(sections))
    return sections


# ── Markdown / TXT ────────────────────────────────────────────────────────────

def _parse_text(file_bytes: bytes, filename: str) -> str:
    try:
        return _clean_text(file_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise MarkdownParseError(f"Failed to read '{filename}': {exc}") from exc


def _parse_text_sections(file_bytes: bytes, filename: str) -> list[str]:
    """Split markdown/text by top-level headings (## or ---) for page hints."""
    try:
        raw = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise MarkdownParseError(f"Failed to read '{filename}': {exc}") from exc

    # Split on markdown H1/H2 headings or horizontal rules
    sections = re.split(r"(?m)^#{1,2}\s+.+$|^---+$", raw)
    sections = [_clean_text(s) for s in sections if s.strip()]

    # If no headings found, split by paragraph
    if len(sections) <= 1:
        sections = [_clean_text(p) for p in re.split(r"\n{2,}", raw) if p.strip()]

    logger.info("'%s': extracted %d sections from markdown/text.", filename, len(sections))
    return sections if sections else [_clean_text(raw)]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise whitespace, remove PDF artefacts, preserve sentence structure."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
