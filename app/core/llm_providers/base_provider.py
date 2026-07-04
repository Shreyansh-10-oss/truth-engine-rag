"""
core/llm_providers/base_provider.py — Abstract Base Class for all LLM providers.
Every provider MUST implement generate() and health_check().
Strategy Pattern: the RAG engine never knows which concrete provider it's using.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Unified interface contract for all LLM backends.
    The llm_router depends ONLY on this interface — never on concrete providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name for logging and response metadata."""
        ...

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        """
        Generate a grounded answer from the prompt and retrieved context.

        Args:
            prompt:  The user's question.
            context: Retrieved document chunks, pre-assembled as a single string.

        Returns:
            The model's answer as a plain string.

        Raises:
            LLMRateLimitError:          HTTP 429 received.
            LLMProviderUnavailableError: HTTP 5xx or connection failure.
            LLMTimeoutError:            Response took longer than configured timeout.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Probe whether the provider is reachable and operational.

        Returns:
            True if the provider can accept requests; False otherwise.
        """
        ...

    # ── Shared prompt assembly ────────────────────────────────────────────────
    # All providers use the same system prompt template (blueprint spec).

    SYSTEM_PROMPT = (
        "You are a document assistant. Answer ONLY from the provided context.\n"
        "If the answer is not in the context, say \"I don't know.\"\n"
        "Be concise and cite relevant details from the context."
    )

    def _build_full_prompt(self, prompt: str, context: str) -> str:
        """
        Assemble the final prompt string following the architecture blueprint template.
        Providers that use a chat-completion API should override this to use
        system/user message roles instead.
        """
        return (
            f"SYSTEM: {self.SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {prompt}"
        )
