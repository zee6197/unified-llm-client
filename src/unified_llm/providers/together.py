"""Together AI provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

import httpx

from unified_llm.errors import ProviderError
from unified_llm.providers.base import BaseProvider, ModelCapabilities
from unified_llm.types import ChatRequest, ChatResponse, Message, StreamEvent

_DEFAULT_BASE_URL = "https://api.together.xyz"
_CHAT_PATH = "/v1/chat/completions"


class TogetherProvider(BaseProvider):
    """Minimal async wrapper for Together chat completions."""

    name = "together"

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
        self._supported_tools: tuple[str, ...] = tuple(supported_tools or ())

    async def aclose(self) -> None:
        await self._client.aclose()

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            tools=self._supported_tools,
            streaming=True,
            thinking=False,
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        payload = self._build_payload(req)
        response = await self._client.post(_CHAT_PATH, headers=self._headers, json=payload)
        data = self._json_or_error(response)

        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content") or ""

        return ChatResponse(provider=self.name, model=req.model, text=text, raw=data)

    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:

        async def _gen() -> AsyncIterator[StreamEvent]:
            payload = self._build_payload(req)
            payload["stream"] = True

            async with self._client.stream("POST", _CHAT_PATH, headers=self._headers, json=payload) as response:
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

                    # Some servers send "event:" lines. Ignore them.
                    if line.startswith("event:"):
                        continue

                    if not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        yield StreamEvent(type="done")
                        return

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
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
        return payload

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _json_or_error(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise ProviderError(
                "together",
                response.text or response.reason_phrase,
                status_code=response.status_code,
            )
        return response.json()

    @staticmethod
    def _extract_delta_text(event: dict[str, Any]) -> str:
        # Common streaming format: choices[0].delta.content
        choices = event.get("choices", [])
        if not choices:
            return ""
        choice0 = choices[0]
        delta = choice0.get("delta", {})
        content = delta.get("content")
        if isinstance(content, str):
            return content

        # Fallbacks (some payloads differ)
        msg = choice0.get("message", {})
        text = msg.get("content")
        return text if isinstance(text, str) else ""
