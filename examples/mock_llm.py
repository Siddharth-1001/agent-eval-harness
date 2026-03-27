"""Shared mock LLM for all agent-eval examples. No API key required."""

from __future__ import annotations

from typing import Any


class MockLLMResponse:
    """Minimal response object that looks like a real LLM response."""

    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.model = "mock-llm-v1"
        self.usage = MockUsage()


class MockUsage:
    input_tokens: int = 50
    output_tokens: int = 30
    prompt_tokens: int = 50
    completion_tokens: int = 30


class MockLLM:
    """
    Deterministic mock LLM that cycles through canned responses.
    Use this in examples so they run offline without any API key.
    """

    model = "mock-llm-v1"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or [
            "I'll search for that information.",
            "Based on my research, here is what I found.",
            "The task is complete. Here is a summary.",
        ]
        self._index = 0

    def invoke(self, messages: Any) -> MockLLMResponse:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return MockLLMResponse(content=response)

    async def ainvoke(self, messages: Any) -> MockLLMResponse:
        return self.invoke(messages)

    def __call__(self, messages: Any) -> MockLLMResponse:
        return self.invoke(messages)
