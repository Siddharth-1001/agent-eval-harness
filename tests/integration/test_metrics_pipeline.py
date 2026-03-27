"""Integration tests for compute_all_metrics pipeline."""

from __future__ import annotations

import pytest

from agent_eval.metrics import (
    CostMetrics,
    HallucinationMetrics,
    LatencyMetrics,
    MetricsConfig,
    MetricsReport,
    ToolMetrics,
    compute_all_metrics,
)
from agent_eval.tracer.schema import TokenCount, ToolCall, Trace, Turn


def make_tool_call(tool_name: str, success: bool, latency_ms: int = 100) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        input_args={"query": "test"},
        success=success,
        latency_ms=latency_ms,
    )


def make_turn(
    turn_id: int, latency_ms: int = 200, prompt_tokens: int = 100, completion_tokens: int = 50
) -> Turn:
    return Turn(
        turn_id=turn_id,
        role="assistant",
        content="assistant response",
        latency_ms=latency_ms,
        tokens=TokenCount(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        tool_calls=[
            make_tool_call("search", success=True, latency_ms=80),
            make_tool_call("fetch", success=False, latency_ms=120),
        ],
    )


def make_full_trace() -> Trace:
    return Trace(
        model="claude-sonnet-4-6",
        task="integration-test",
        turns=[
            make_turn(0, latency_ms=200, prompt_tokens=500, completion_tokens=200),
            make_turn(1, latency_ms=400, prompt_tokens=600, completion_tokens=300),
            make_turn(2, latency_ms=300, prompt_tokens=400, completion_tokens=150),
        ],
    )


@pytest.mark.asyncio
async def test_full_trace_produces_populated_metrics_report():
    trace = make_full_trace()
    report = await compute_all_metrics(trace)

    assert isinstance(report, MetricsReport)
    assert isinstance(report.tool, ToolMetrics)
    assert isinstance(report.hallucination, HallucinationMetrics)
    assert isinstance(report.latency, LatencyMetrics)
    assert isinstance(report.cost, CostMetrics)


@pytest.mark.asyncio
async def test_all_sub_metrics_are_present_and_typed_correctly():
    trace = make_full_trace()
    report = await compute_all_metrics(trace)

    # Tool metrics
    assert report.tool.total_calls == 6  # 3 turns * 2 calls each
    assert report.tool.successful_calls == 3
    assert report.tool.failed_calls == 3
    assert report.tool.success_rate == pytest.approx(0.5)

    # Latency metrics
    assert report.latency.total_ms == 900  # 200 + 400 + 300
    assert report.latency.slowest_turn_id == 1  # turn index 1 has latency 400
    assert report.latency.p50_ms == 300
    assert isinstance(report.latency.tool_latency_breakdown, list)

    # Cost metrics
    assert report.cost.model == "claude-sonnet-4-6"
    assert report.cost.input_tokens == 1500  # 500 + 600 + 400
    assert report.cost.output_tokens == 650  # 200 + 300 + 150
    assert report.cost.total_usd > 0
    assert len(report.cost.cost_per_turn) == 3

    # Hallucination metrics
    assert (
        report.hallucination.total_flags == 0
    )  # no schemas configured, default mode has no schema
    assert report.hallucination.hallucination_rate == 0.0


@pytest.mark.asyncio
async def test_compute_all_metrics_with_custom_config():
    trace = make_full_trace()
    config = MetricsConfig(
        pricing_overrides={"claude-sonnet-4-6": {"input_per_1m": 1.0, "output_per_1m": 1.0}}
    )
    report = await compute_all_metrics(trace, config)
    # 1500 input + 650 output = 2150 tokens * $1/1M = $0.00215
    assert report.cost.total_usd == pytest.approx(2150 / 1_000_000)


@pytest.mark.asyncio
async def test_empty_trace_pipeline():
    trace = Trace(model="gpt-4o", turns=[])
    report = await compute_all_metrics(trace)

    assert report.tool.total_calls == 0
    assert report.tool.success_rate == 0.0
    assert report.latency.total_ms == 0
    assert report.latency.slowest_turn_id is None
    assert report.cost.total_usd == 0.0
    assert report.hallucination.total_flags == 0


@pytest.mark.asyncio
async def test_metrics_report_model_fields():
    """Verify MetricsReport fields have correct types from pydantic."""
    trace = make_full_trace()
    report = await compute_all_metrics(trace)

    # Pydantic model_fields should be accessible
    assert "tool" in MetricsReport.model_fields
    assert "hallucination" in MetricsReport.model_fields
    assert "latency" in MetricsReport.model_fields
    assert "cost" in MetricsReport.model_fields

    # Verify report serializes without error
    d = report.model_dump()
    assert "tool" in d
    assert "hallucination" in d
    assert "latency" in d
    assert "cost" in d
