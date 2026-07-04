"""
core/llm_router.py — LLM Fallback Chain
Responsibility: Try selected provider → Secondary → Ollama → raise FatalLLMError.
The RAG engine calls only this router — never a provider directly.

Fallback chain (blueprint spec):
    Primary (user-selected) → Groq → Ollama → FatalLLMError
"""

import logging
from dataclasses import dataclass

from app.core.llm_providers.base_provider import BaseLLMProvider
from app.core.llm_providers.gemini_provider import GeminiProvider
from app.core.llm_providers.groq_provider import GroqProvider
from app.core.llm_providers.ollama_provider import OllamaProvider
from app.utils.exceptions import (
    FatalLLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# Exceptions that trigger a fallback (blueprint: 429, 5xx, timeout, generic)
_FALLBACK_EXCEPTIONS = (
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
    LLMProviderError,
)


@dataclass
class RouterResult:
    answer: str
    provider_used: str
    fallback_triggered: bool


class LLMRouter:
    """
    Manages the provider registry and executes the fallback chain.

    Instantiated once at FastAPI startup and shared across requests.
    """

    PROVIDER_NAMES = ["gemini", "groq", "ollama"]

    def __init__(
        self,
        gemini_provider: GeminiProvider,
        groq_provider: GroqProvider,
        ollama_provider: OllamaProvider,
    ) -> None:
        self._providers: dict[str, BaseLLMProvider] = {
            "gemini": gemini_provider,
            "groq": groq_provider,
            "ollama": ollama_provider,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        context: str,
        preferred_provider: str = "gemini",
    ) -> RouterResult:
        """
        Generate an answer, automatically falling back through the chain on failure.

        Args:
            prompt:             User question.
            context:            Retrieved chunk text.
            preferred_provider: One of 'gemini', 'groq', 'ollama' — user selection.

        Returns:
            RouterResult with answer, provider_used, and fallback_triggered flag.

        Raises:
            FatalLLMError: All providers in the chain have failed.
        """
        chain = self._build_chain(preferred_provider)
        errors: list[str] = []
        fallback_triggered = False

        for i, provider in enumerate(chain):
            try:
                logger.info("LLMRouter: trying provider '%s'.", provider.provider_name)
                answer = provider.generate(prompt, context)
                if i > 0:
                    fallback_triggered = True
                    logger.info(
                        "LLMRouter: fallback succeeded with '%s' (original: '%s').",
                        provider.provider_name, preferred_provider,
                    )
                return RouterResult(
                    answer=answer,
                    provider_used=provider.provider_name,
                    fallback_triggered=fallback_triggered,
                )
            except _FALLBACK_EXCEPTIONS as exc:
                errors.append(f"{provider.provider_name}: {exc}")
                logger.warning(
                    "LLMRouter: provider '%s' failed (%s). Trying next in chain.",
                    provider.provider_name, exc,
                )
                continue
            except Exception as exc:
                errors.append(f"{provider.provider_name}: unexpected — {exc}")
                logger.error(
                    "LLMRouter: unexpected error from '%s': %s.",
                    provider.provider_name, exc,
                )
                continue

        # All providers exhausted
        logger.error("LLMRouter: ALL providers failed. Errors: %s", errors)
        raise FatalLLMError(
            "All LLM providers are currently unavailable. Details: " + " | ".join(errors)
        )

    def health_status(self) -> dict[str, bool]:
        """Return health check results for all registered providers."""
        return {name: p.health_check() for name, p in self._providers.items()}

    def get_provider(self, name: str) -> BaseLLMProvider | None:
        return self._providers.get(name)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_chain(self, preferred: str) -> list[BaseLLMProvider]:
        """
        Build the ordered fallback list.
        Preferred goes first; remaining providers fill in without duplicating.
        Blueprint order: Gemini → Groq → Ollama.
        """
        ordered_defaults = ["gemini", "groq", "ollama"]
        chain_names: list[str] = [preferred] + [
            p for p in ordered_defaults if p != preferred
        ]
        return [self._providers[n] for n in chain_names if n in self._providers]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_router_from_config() -> LLMRouter:
    """Construct LLMRouter using the global config singleton."""
    from config import config
    return LLMRouter(
        gemini_provider=GeminiProvider(
            api_key=config.llm.gemini_api_key,
            model=config.llm.gemini_model,
            timeout=config.llm.request_timeout,
        ),
        groq_provider=GroqProvider(
            api_key=config.llm.groq_api_key,
            model=config.llm.groq_model,
            timeout=config.llm.request_timeout,
        ),
        ollama_provider=OllamaProvider(
            base_url=config.llm.ollama_base_url,
            model=config.llm.ollama_model,
            timeout=config.llm.request_timeout,
        ),
    )
