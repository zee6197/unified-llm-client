import asyncio

from unified_llm.client import LLMClient
from unified_llm.providers.anthropic import AnthropicProvider
from unified_llm.providers.openai import OpenAIProvider
from unified_llm.providers.together import TogetherProvider
from unified_llm.types import ChatRequest, Message, ToolDef


async def main() -> None:
    client = LLMClient(
        openai=OpenAIProvider(api_key="DUMMY"),
        anthropic=AnthropicProvider(api_key="DUMMY"),
        together=TogetherProvider(api_key="DUMMY"),
    )

    # Demonstrate capability gating (Together doesn't support tools)
    req = ChatRequest(
        provider="together",
        model="any",
        messages=[Message(role="user", content="hi")],
        tools=[ToolDef(name="demo_tool")],
        tool_mode="auto",
    )

    try:
        await client.chat(req)
    except Exception as e:
        print("Expected error:", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())
