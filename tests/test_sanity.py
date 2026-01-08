import asyncio
import unittest
from collections.abc import AsyncIterator

from unified_llm.client import LLMClient
from unified_llm.errors import ToolNotAvailableError
from unified_llm.providers.base import BaseProvider, ModelCapabilities, ensure_capabilities
from unified_llm.types import ChatRequest, ChatResponse, Message, StreamEvent, ToolDef


class DummyProvider(BaseProvider):
    name = "dummy"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(tools=("*",), streaming=True, thinking=True)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(provider=self.name, model=req.model, text="ok", raw={"text": "ok"})

    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        async def _gen() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type="text_delta", text="chunk", raw={"delta": "chunk"})
            yield StreamEvent(type="done", raw={"done": True})

        return _gen()


class ClientSanityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.req = ChatRequest(
            provider="dummy",
            model="toy",
            messages=[Message(role="user", content="hi")],
        )

    def test_ensure_capabilities_blocks_tools(self) -> None:
        req = self.req.model_copy(update={"tools": [ToolDef(name="math")], "tool_mode": "auto"})
        caps = ModelCapabilities(tools=(), streaming=True, thinking=True)
        with self.assertRaises(ToolNotAvailableError):
            ensure_capabilities(req, caps)

    def test_client_stream_returns_async_iterator(self) -> None:
        provider = DummyProvider()
        client = LLMClient(openai=provider)

        resp = asyncio.run(client.chat(self.req))
        self.assertEqual(resp.text, "ok")

        stream_iter = client.stream(self.req)
        self.assertIsInstance(stream_iter, AsyncIterator)
        events = asyncio.run(_collect_events(stream_iter))
        self.assertEqual([e.type for e in events], ["text_delta", "done"])


async def _collect_events(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
    return events


if __name__ == "__main__":
    unittest.main()
