"""Tests for LatencyAnalyzer."""

from __future__ import annotations

import pytest

from agent_eval.metrics.latency import LatencyAnalyzer
from agent_eval.tracer.schema import ToolCall, Trace, Turn


def make_trace(turns: list[Turn]) -> Trace:
    return Trace(model="claude-sonnet-4-6", turns=turns)


def make_turn(turn_id: int, latency_ms: int, tool_calls: list[ToolCall] | None = None) -> Turn:
    return Turn(
        turn_id=turn_id,
        role="assistant",
        content="",
        latency_ms=latency_ms,
        tool_calls=tool_calls or [],
    )


def make_call(tool_name: str, latency_ms: int) -> ToolCall:
    return ToolCall(tool_name=tool_name, input_args={}, success=True, latency_ms=latency_ms)


@pytest.mark.asyncio
async def test_empty_trace_returns_zeros():
    trace = make_trace([])
    result = await LatencyAnalyzer().analyze(trace)
    assert result.total_ms == 0
    assert result.p50_ms == 0
    assert result.p95_ms == 0
    assert result.slowest_turn_id is None
    assert result.tool_latency_breakdown == []


@pytest.mark.asyncio
async def test_p50_computed_correctly():
    # Latencies: [100, 200, 300] -> median = 200
    turns = [
        make_turn(0, 100),
        make_turn(1, 200),
        make_turn(2, 300),
    ]
    trace = make_trace(turns)
    result = await LatencyAnalyzer().analyze(trace)
    assert result.p50_ms == 200
    assert result.total_ms == 600


@pytest.mark.asyncio
async def test_p95_computed_correctly():
    # 20 turns with latencies 10, 20, ..., 200
    turns = [make_turn(i, (i + 1) * 10) for i in range(20)]
    trace = make_trace(turns)
    result = await LatencyAnalyzer().analyze(trace)
    # p95 index = max(0, int(20 * 0.95) - 1) = max(0, 19 - 1) = 18 -> latency = 190
    assert result.p95_ms == 190


@pytest.mark.asyncio
async def test_slowest_turn_id_correctly_identified():
    turns = [
        make_turn(0, 100),
        make_turn(1, 500),
        make_turn(2, 200),
    ]
    trace = make_trace(turns)
    result = await LatencyAnalyzer().analyze(trace)
    assert result.slowest_turn_id == 1


@pytest.mark.asyncio
async def test_tool_latency_breakdown_computed():
    calls = [
        make_call("search", 100),
        make_call("search", 200),
        make_call("fetch", 300),
    ]
    turns = [make_turn(0, 400, calls)]
    trace = make_trace(turns)
    result = await LatencyAnalyzer().analyze(trace)

    by_tool = {t.tool_name: t for t in result.tool_latency_breakdown}
    assert "search" in by_tool
    assert by_tool["search"].avg_latency_ms == pytest.approx(150.0)
    assert by_tool["search"].call_count == 2

    assert "fetch" in by_tool
    assert by_tool["fetch"].avg_latency_ms == pytest.approx(300.0)
    assert by_tool["fetch"].call_count == 1


@pytest.mark.asyncio
async def test_single_turn_p50_and_p95():
    trace = make_trace([make_turn(0, 250)])
    result = await LatencyAnalyzer().analyze(trace)
    assert result.p50_ms == 250
    assert result.p95_ms == 250
    assert result.slowest_turn_id == 0
