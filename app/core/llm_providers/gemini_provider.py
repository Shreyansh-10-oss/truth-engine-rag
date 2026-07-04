"""
core/llm_providers/gemini_provider.py — Google Gemini provider.
Uses threading.Thread for timeout — cross-platform, works on Windows.
"""

import logging
import threading

from app.core.llm_providers.base_provider import BaseLLMProvider
from app.utils.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", timeout: int = 15) -> None:
        self._api_key = api_key
        self._model_name = model
        self._timeout = timeout
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self._model_name)
            except Exception as exc:
                raise LLMProviderUnavailableError(self.provider_name, f"Client init failed: {exc}") from exc
        return self._client

    def generate(self, prompt: str, context: str) -> str:
        client = self._get_client()
        full_prompt = self._build_full_prompt(prompt, context)
        result = {"response": None, "error": None}

        def _call():
            try:
                result["response"] = client.generate_content(full_prompt)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            raise LLMTimeoutError(self.provider_name, f"Timed out after {self._timeout}s")

        if result["error"] is not None:
            exc = result["error"]
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str or "rate" in exc_str:
                raise LLMRateLimitError(self.provider_name, str(exc))
            if "500" in exc_str or "503" in exc_str or "unavailable" in exc_str:
                raise LLMProviderUnavailableError(self.provider_name, str(exc))
            raise LLMProviderUnavailableError(self.provider_name, f"Unexpected error: {exc}")

        try:
            return result["response"].text.strip()
        except Exception as exc:
            raise LLMProviderError(self.provider_name, f"Could not parse response: {exc}") from exc

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.generate_content("Reply with the single word: OK")
            return bool(resp.text)
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False
