# unified-llm

A small async Python client that exposes a single interface over a few LLM providers
(OpenAI, Anthropic/Claude, and Together).

The main goal is to make provider differences explicit and predictable:
- same request shape
- same response shape
- clear errors when a feature is not supported

Streaming is implemented for OpenAI and Together; Anthropic is intentionally chat-only in this MVP to keep scope small.

This avoids scattering provider-specific logic throughout application code.


## Install (dev)
```bash
python3 -m pip install -e .
```

Use whichever Python interpreter you prefer (the project targets Python 3.10+); just swap `python3` for the appropriate command in your environment.

## Quick start

### Create the client
```python
from unified_llm.client import LLMClient
from unified_llm.providers.openai import OpenAIProvider
from unified_llm.providers.anthropic import AnthropicProvider
from unified_llm.providers.together import TogetherProvider

client = LLMClient(
    openai=OpenAIProvider(api_key="OPENAI_KEY"),
    anthropic=AnthropicProvider(api_key="ANTHROPIC_KEY"),
    together=TogetherProvider(api_key="TOGETHER_KEY"),
)
```

### Non-streaming chat
```python
from unified_llm.types import ChatRequest, Message

req = ChatRequest(
    provider="openai",
    model="gpt-4o-mini",
    messages=[Message(role="user", content="Say hello in three words.")],
)

resp = await client.chat(req)
print(resp.text)
```

### Streaming
```python
from unified_llm.types import ChatRequest, Message

req = ChatRequest(
    provider="openai",
    model="gpt-4o-mini",
    messages=[Message(role="user", content="Stream a short sentence.")],
)

async for event in client.stream(req):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    if event.type == "done":
        print()
```

## Provider support (current MVP)

| Provider  | Chat | Streaming | Tools | Thinking |
|-----------|------|-----------|-------|----------|
| OpenAI    | Yes  | Yes       | Yes   | Yes      |
| Anthropic | Yes  | No*       | Yes   | Yes      |
| Together  | Yes  | Yes       | No    | No       |

*Streaming is left unimplemented for the Anthropic adapter to keep this MVP small.

Unsupported features fail fast with a clear error.

### Capability guard example

```python
from unified_llm.client import LLMClient
from unified_llm.providers.together import TogetherProvider
from unified_llm.types import ChatRequest, Message
from unified_llm.errors import ToolNotAvailableError
from unified_llm.types import ToolDef

client = LLMClient(together=TogetherProvider(api_key="TOGETHER_KEY"))

req = ChatRequest(
    provider="together",
    model="any",
    messages=[Message(role="user", content="hi")],
    tools=[ToolDef(name="demo_tool")],
    tool_mode="auto",  # tools not supported by Together in this MVP
)

try:
    await client.chat(req)
except ToolNotAvailableError as exc:
    print("Capability check blocked the request:", exc)
```

Each provider advertises a list of tool names (or `"*"` for "any tool"). Requests that reference tool names outside that list fail before any network call is made.

## How it works (high level)

1. You construct an `LLMClient` with one or more providers.
2. You send a normalized `ChatRequest` that includes the provider name.
3. The client checks the provider’s declared capabilities.
4. If the request is allowed, the provider adapter makes the HTTP call.
5. Responses are returned as a normalized `ChatResponse` or streamed as `StreamEvent` objects.

## Errors

- `UnsupportedProviderError`: client was not configured with the requested provider.
- `UnsupportedFeatureError`: tools, streaming, or thinking were requested but not supported.
- `ToolNotAvailableError`: provider does not advertise one or more of the requested tool names.
- `ProviderError`: HTTP or API errors from a provider (status code included when available).
# test
