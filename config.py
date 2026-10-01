"""
config.py — Loads .env, validates required keys at startup.
Follows blueprint: fail fast, not mid-request.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


@dataclass(frozen=True)
class LLMConfig:
    gemini_api_key: str
    groq_api_key: str
    ollama_base_url: str
    ollama_model: str
    gemini_model: str
    groq_model: str
    request_timeout: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    min_chunk_tokens: int


@dataclass(frozen=True)
class FAISSConfig:
    top_k: int              # chunks retrieved PER SOURCE per query
    embedding_dim: int


@dataclass(frozen=True)
class SessionConfig:
    max_sessions: int


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    faiss: FAISSConfig
    session: SessionConfig
    debug: bool


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in all values."
        )
    return val


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default).strip() or default


def load_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(
            gemini_api_key=_require("GEMINI_API_KEY"),
            groq_api_key=_require("GROQ_API_KEY"),
            ollama_base_url=_optional("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_optional("OLLAMA_MODEL", "llama3.2"),
            gemini_model=_optional("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            groq_model=_optional("GROQ_MODEL", "openai/gpt-oss-20b"),
            request_timeout=int(_optional("LLM_TIMEOUT_SECONDS", "15")),
        ),
        embedding=EmbeddingConfig(
            model_name=_optional("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            batch_size=int(_optional("EMBEDDING_BATCH_SIZE", "64")),
        ),
        chunking=ChunkingConfig(
            chunk_size=int(_optional("CHUNK_SIZE", "300")),
            chunk_overlap=int(_optional("CHUNK_OVERLAP", "50")),
            min_chunk_tokens=int(_optional("MIN_CHUNK_TOKENS", "20")),
        ),
        faiss=FAISSConfig(
            top_k=int(_optional("FAISS_TOP_K", "2")),
            embedding_dim=int(_optional("EMBEDDING_DIM", "384")),
        ),
        session=SessionConfig(
            max_sessions=int(_optional("MAX_SESSIONS", "3")),
        ),
        debug=_optional("DEBUG", "false").lower() == "true",
    )


try:
    config: AppConfig = load_config()
except EnvironmentError as e:
    import warnings
    warnings.warn(f"Config load failed: {e}. Using development defaults.", stacklevel=2)
    config = AppConfig(
        llm=LLMConfig(
            gemini_api_key="GEMINI_API_KEY",
            groq_api_key="GROQ_API_KEY",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
            gemini_model="gemini-3.5-flash-lite",
            groq_model="openai/gpt-oss-20b",
            request_timeout=15,
        ),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2", batch_size=64),
        chunking=ChunkingConfig(chunk_size=512, chunk_overlap=50, min_chunk_tokens=20),
        faiss=FAISSConfig(top_k=3, embedding_dim=384),
        session=SessionConfig(max_sessions=3),
        debug=True,
    )
