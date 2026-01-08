"""Minimal provider-agnostic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolMode = Literal["off", "auto", "required"]
ThinkingMode = Literal["off", "on"]


class Message(BaseModel):
    """Single chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ToolDef(BaseModel):
    """Simple JSON-schema tool definition."""

    name: str
    description: str | None = None
    json_schema: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Normalized request shared by all providers."""

    provider: str
    model: str
    messages: list[Message]
    tools: list[ToolDef] = Field(default_factory=list)
    tool_mode: ToolMode = "off"
    thinking: ThinkingMode = "off"
    temperature: float | None = None
    max_tokens: int | None = None


class ChatResponse(BaseModel):
    """Simplified chat response."""

    provider: str
    model: str
    text: str
    # provider-specific payload kept for debugging or advanced use
    raw: dict[str, Any]


class StreamEvent(BaseModel):
    """Streaming chunks emitted by providers."""

    type: Literal["text_delta", "done"]
    text: str | None = None
    raw: dict[str, Any] | None = None
