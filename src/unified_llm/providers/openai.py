"""OpenAI provider implementation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

import httpx

from unified_llm.errors import ProviderError
from unified_llm.providers.base import BaseProvider, ModelCapabilities
from unified_llm.types import ChatRequest, ChatResponse, Message, StreamEvent, ToolDef

_DEFAULT_BASE_URL = "https://api.openai.com"
_CHAT_PATH = "/v1/chat/completions"


class OpenAIProvider(BaseProvider):
    """Minimal async wrapper for the OpenAI Chat Completions API."""

    name = "openai"
    _logger = logging.getLogger(__name__)

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
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._supported_tools: tuple[str, ...] = tuple(supported_tools or ("*",))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def capabilities(self, model: str) -> ModelCapabilities:
        """Return capabilities for OpenAI chat models."""
        return ModelCapabilities(
            tools=self._supported_tools,
            streaming=True,
            thinking=True,
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Call OpenAI Chat Completions and normalize the result."""
        payload = self._build_payload(req)
        response = await self._client.post(_CHAT_PATH, headers=self._headers, json=payload)
        data = self._json_or_error(response)

        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content") or ""

        return ChatResponse(provider=self.name, model=req.model, text=text, raw=data)

    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Return an async iterator that streams text deltas."""

        async def _gen() -> AsyncIterator[StreamEvent]:
            payload = self._build_payload(req)
            payload["stream"] = True

            async with self._client.stream(
                "POST",
                _CHAT_PATH,
                headers=self._headers,
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise ProviderError(
                        self.name,
                        body.decode() or response.reason_phrase,
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()

                    # OpenAI streaming uses SSE. We only care about "data:" lines.
                    if not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        yield StreamEvent(type="done")
                        return

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        self._logger.debug("Skipping non-JSON streaming chunk: %s", data_str)
                        continue

                    chunk = self._extract_delta_text(event)
                    if chunk:
                        yield StreamEvent(type="text_delta", text=chunk, raw=event)

        return _gen()

    def _build_payload(self, req: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [self._serialize_message(m) for m in req.messages],
        }

        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if req.tool_mode != "off" and req.tools:
            payload.update(self._serialize_tools(req.tools, req.tool_mode))

        # Note: thinking is modeled at the unified layer; provider-specific toggles can be added here.
        return payload

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _serialize_tools(tools: list[ToolDef], tool_mode: str) -> dict[str, Any]:
        tool_payload = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.json_schema,
                },
            }
            for t in tools
        ]

        payload: dict[str, Any] = {"tools": tool_payload}

        if tool_mode == "auto":
            payload["tool_choice"] = "auto"
        elif tool_mode == "required":
            # MVP: require the first tool
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tools[0].name},
            }

        return payload

    @staticmethod
    def _json_or_error(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise ProviderError(
                "openai",
                response.text or response.reason_phrase,
                status_code=response.status_code,
            )
        return cast(dict[str, Any], response.json())

    @staticmethod
    def _extract_delta_text(event: dict[str, Any]) -> str:
        """Extract the standard OpenAI streaming text delta (delta.content)."""
        choices = event.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        return content if isinstance(content, str) else ""
