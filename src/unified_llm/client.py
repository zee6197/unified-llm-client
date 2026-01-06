"""Async client orchestrating provider interactions."""

from __future__ import annotations

from typing import AsyncIterator

from unified_llm.errors import UnsupportedFeatureError, UnsupportedProviderError
from unified_llm.providers.base import BaseProvider, ModelCapabilities, ensure_capabilities
from unified_llm.types import ChatRequest, ChatResponse, StreamEvent


class LLMClient:
    """High-level coordinator for chatting with configured providers."""

    def __init__(
        self,
        *,
        openai: BaseProvider | None = None,
        anthropic: BaseProvider | None = None,
        together: BaseProvider | None = None,
    ) -> None:
        providers = (openai, anthropic, together)
        self._providers: dict[str, BaseProvider] = {}
        for provider in providers:
            if provider is not None:
                self._providers[provider.name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Return a provider by its registered name."""
        try:
            return self._providers[name]
        except KeyError as exc:
            raise UnsupportedProviderError(name) from exc

    def capabilities(self, provider: str, model: str) -> ModelCapabilities:
        """Return model capability info for a provider."""
        return self.get_provider(provider).capabilities(model)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Execute an async chat completion request."""
        provider = self.get_provider(req.provider)
        caps = provider.capabilities(req.model)
        ensure_capabilities(req, caps)
        return await provider.chat(req)

    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream events for a chat request."""
        provider = self.get_provider(req.provider)
        caps = provider.capabilities(req.model)
        if not caps.streaming:
            raise UnsupportedFeatureError("streaming")
        ensure_capabilities(req, caps)
        return provider.stream(req)
