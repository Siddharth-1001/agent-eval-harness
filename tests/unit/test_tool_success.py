"""Tests for ToolSuccessAnalyzer."""

from __future__ import annotations

import pytest

from agent_eval.metrics.tool_success import ToolSuccessAnalyzer
from agent_eval.tracer.schema import ToolCall, Trace, Turn


def make_trace(turns: list[Turn]) -> Trace:
    return Trace(model="claude-sonnet-4-6", turns=turns)


def make_turn(turn_id: int, tool_calls: list[ToolCall]) -> Turn:
    return Turn(turn_id=turn_id, role="assistant", content="", tool_calls=tool_calls)


def make_call(tool_name: str, success: bool) -> ToolCall:
    return ToolCall(tool_name=tool_name, input_args={}, success=success, latency_ms=100)


@pytest.mark.asyncio
async def test_empty_trace_returns_zero_metrics():
    trace = make_trace([])
    result = await ToolSuccessAnalyzer().analyze(trace)
    assert result.total_calls == 0
    assert result.successful_calls == 0
    assert result.failed_calls == 0
    assert result.success_rate == 0.0
    assert result.per_tool == []


@pytest.mark.asyncio
async def test_all_success_returns_100_percent():
    calls = [make_call("search", True), make_call("search", True), make_call("fetch", True)]
    trace = make_trace([make_turn(0, calls)])
    result = await ToolSuccessAnalyzer().analyze(trace)
    assert result.total_calls == 3
    assert result.successful_calls == 3
    assert result.failed_calls == 0
    assert result.success_rate == 1.0


@pytest.mark.asyncio
async def test_mixed_calls_returns_correct_rate():
    calls = [
        make_call("search", True),
        make_call("search", False),
        make_call("fetch", True),
        make_call("fetch", False),
        make_call("fetch", False),
    ]
    trace = make_trace([make_turn(0, calls)])
    result = await ToolSuccessAnalyzer().analyze(trace)
    assert result.total_calls == 5
    assert result.successful_calls == 2
    assert result.failed_calls == 3
    assert result.success_rate == pytest.approx(2 / 5)


@pytest.mark.asyncio
async def test_per_tool_breakdown_is_accurate():
    calls = [
        make_call("search", True),
        make_call("search", False),
        make_call("fetch", True),
        make_call("fetch", True),
    ]
    trace = make_trace([make_turn(0, calls)])
    result = await ToolSuccessAnalyzer().analyze(trace)

    by_tool = {b.tool_name: b for b in result.per_tool}

    assert "search" in by_tool
    assert by_tool["search"].total_calls == 2
    assert by_tool["search"].successful_calls == 1
    assert by_tool["search"].failed_calls == 1
    assert by_tool["search"].success_rate == pytest.approx(0.5)

    assert "fetch" in by_tool
    assert by_tool["fetch"].total_calls == 2
    assert by_tool["fetch"].successful_calls == 2
    assert by_tool["fetch"].failed_calls == 0
    assert by_tool["fetch"].success_rate == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calls_across_multiple_turns():
    turn0 = make_turn(0, [make_call("search", True)])
    turn1 = make_turn(1, [make_call("search", False), make_call("fetch", True)])
    trace = make_trace([turn0, turn1])
    result = await ToolSuccessAnalyzer().analyze(trace)
    assert result.total_calls == 3
    assert result.successful_calls == 2
    assert result.failed_calls == 1
