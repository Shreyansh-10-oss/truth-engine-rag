"""
core/llm_providers/ollama_provider.py — Ollama provider.
Tertiary in the fallback chain: always available locally.
Communicates with Ollama REST API — no external dependencies.
"""

import logging

import requests

from app.core.llm_providers.base_provider import BaseLLMProvider
from app.utils.exceptions import LLMProviderUnavailableError, LLMTimeoutError

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Wraps the Ollama local REST API.
    Requires: Ollama running on localhost:11434 (or configured OLLAMA_BASE_URL).
    No API key needed — local inference.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 15,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, context: str) -> str:
        full_prompt = self._build_full_prompt(prompt, context)
        url = f"{self._base_url}/api/generate"

        payload = {
            "model": self._model_name,
            "prompt": full_prompt,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(self.provider_name, f"Timed out after {self._timeout}s") from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMProviderUnavailableError(
                self.provider_name,
                f"Cannot reach Ollama at {self._base_url}. Is it running?"
            ) from exc

        if resp.status_code != 200:
            raise LLMProviderUnavailableError(
                self.provider_name,
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as exc:
            raise LLMProviderUnavailableError(
                self.provider_name, f"Could not parse Ollama response: {exc}"
            ) from exc

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
