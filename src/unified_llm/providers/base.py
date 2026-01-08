"""Provider-agnostic base interfaces and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from unified_llm.errors import ToolNotAvailableError, UnsupportedFeatureError
from unified_llm.types import ChatRequest, ChatResponse, StreamEvent


@dataclass(frozen=True)
class ModelCapabilities:
    """Describes feature support for a provider model."""

    tools: tuple[str, ...]
    streaming: bool
    thinking: bool


class BaseProvider(ABC):
    """Abstract base class for provider implementations."""

    name: str

    @abstractmethod
    def capabilities(self, model: str) -> ModelCapabilities:
        """Return capability flags for the given model identifier."""
        raise NotImplementedError

    @abstractmethod
    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Execute an async chat completion request."""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Yield streaming events for the request."""
        raise NotImplementedError


def ensure_capabilities(req: ChatRequest, caps: ModelCapabilities) -> None:
    """Fail fast if the request asks for unsupported features."""

    if req.tool_mode != "off" and req.tools:
        if not caps.tools:
            raise ToolNotAvailableError(req.provider, [tool.name for tool in req.tools])
        allowed = set(caps.tools)
        if "*" not in allowed:
            missing = [tool.name for tool in req.tools if tool.name not in allowed]
            if missing:
                raise ToolNotAvailableError(req.provider, missing)

    if req.thinking != "off" and not caps.thinking:
        raise UnsupportedFeatureError("thinking")
