"""Anthropic provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

import httpx

from unified_llm.errors import ProviderError, UnsupportedFeatureError
from unified_llm.providers.base import BaseProvider, ModelCapabilities, ensure_capabilities
from unified_llm.types import ChatRequest, ChatResponse, Message, StreamEvent, ToolDef

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_MESSAGES_PATH = "/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 512


class AnthropicProvider(BaseProvider):
    """Minimal async wrapper for Anthropic Messages API (non-streaming in this MVP)."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        supported_tools: Iterable[str] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url or _DEFAULT_BASE_URL, timeout=timeout_s)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        self._supported_tools: tuple[str, ...] = tuple(supported_tools or ("*",))

    async def aclose(self) -> None:
        await self._client.aclose()

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            tools=self._supported_tools,
            streaming=False,
            thinking=True,
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        ensure_capabilities(req, self.capabilities(req.model))
        payload = self._build_payload(req)
        response = await self._client.post(_MESSAGES_PATH, headers=self._headers, json=payload)
        data = self._json_or_error(response)
        text = self._extract_text(data)
        return ChatResponse(provider=self.name, model=req.model, text=text, raw=data)

    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        ensure_capabilities(req, self.capabilities(req.model))
        raise UnsupportedFeatureError("streaming")

    def _build_payload(self, req: ChatRequest) -> dict[str, Any]:
        system_text, msgs = self._split_system(req.messages)

        payload: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": [self._serialize_message(m) for m in msgs],
        }

        if system_text:
            payload["system"] = system_text
        if req.temperature is not None:
            payload["temperature"] = req.temperature

        if req.tool_mode != "off" and req.tools:
            payload.update(self._serialize_tools(req.tools, req.tool_mode))

        # Note: thinking is modeled at the unified layer; provider-specific toggles can be added here.
        return payload

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        system_parts: list[str] = []
        rest: list[Message] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                rest.append(m)
        return ("\n".join(system_parts), rest)

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        # Anthropic Messages API expects only user/assistant roles here.
        if message.role not in ("user", "assistant"):
            raise ProviderError("anthropic", f"Unsupported role: {message.role}")
        return {
            "role": message.role,
            "content": [{"type": "text", "text": message.content}],
        }

    @staticmethod
    def _serialize_tools(tools: list[ToolDef], tool_mode: str) -> dict[str, Any]:
        payload_tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.json_schema,
            }
            for t in tools
        ]
        payload: dict[str, Any] = {"tools": payload_tools}

        if tool_mode == "auto":
            payload["tool_choice"] = {"type": "auto"}
        elif tool_mode == "required":
            payload["tool_choice"] = {"type": "any"}

        return payload

    @staticmethod
    def _json_or_error(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise ProviderError(
                "anthropic",
                response.text or response.reason_phrase,
                status_code=response.status_code,
            )
        return cast(dict[str, Any], response.json())

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        blocks = data.get("content") or []
        parts: list[str] = []
        for b in blocks:
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
