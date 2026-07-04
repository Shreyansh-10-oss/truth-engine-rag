"""
core/llm_providers/groq_provider.py — Groq provider.
Secondary in the fallback chain: fast inference.
"""

import logging

from app.core.llm_providers.base_provider import BaseLLMProvider
from app.utils.exceptions import (
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    Wraps the Groq Python SDK.
    Requires: GROQ_API_KEY in .env
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", timeout: int = 15) -> None:
        self._api_key = api_key
        self._model_name = model
        self._timeout = timeout
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key, timeout=self._timeout)
            except Exception as exc:
                raise LLMProviderUnavailableError(self.provider_name, f"Client init failed: {exc}") from exc
        return self._client

    def generate(self, prompt: str, context: str) -> str:
        client = self._get_client()

        # Groq uses chat completion — use system/user roles
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
        ]

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate_limit" in exc_str:
                logger.warning("Groq rate limit hit.")
                raise LLMRateLimitError(self.provider_name, str(exc)) from exc
            if "timeout" in exc_str:
                raise LLMTimeoutError(self.provider_name, f"Timed out after {self._timeout}s") from exc
            if any(x in exc_str for x in ["500", "503", "connection"]):
                raise LLMProviderUnavailableError(self.provider_name, str(exc)) from exc
            raise LLMProviderUnavailableError(self.provider_name, f"Unexpected error: {exc}") from exc

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": "Reply with the word OK"}],
                max_tokens=5,
            )
            return bool(resp.choices[0].message.content)
        except Exception as exc:
            logger.warning("Groq health check failed: %s", exc)
            return False
