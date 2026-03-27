"""Shared fixtures for all tests."""

from __future__ import annotations

import pytest

from agent_eval.tracer.schema import TokenCount, ToolCall, Trace, Turn


@pytest.fixture
def sample_tool_call() -> ToolCall:
    return ToolCall(
        tool_name="search_web",
        input_args={"query": "test query"},
        output="some results",
        success=True,
        latency_ms=100,
    )


@pytest.fixture
def sample_turn(sample_tool_call: ToolCall) -> Turn:
    return Turn(
        turn_id=0,
        role="assistant",
        content="I'll search for that",
        tool_calls=[sample_tool_call],
        latency_ms=150,
        tokens=TokenCount(prompt_tokens=50, completion_tokens=20),
    )


@pytest.fixture
def sample_trace(sample_turn: Turn) -> Trace:
    return Trace(
        model="claude-sonnet-4-6",
        task="test task",
        turns=[sample_turn],
    )
