"""Tests for CostCalculator."""

from __future__ import annotations

import pytest

from agent_eval.metrics.cost import CostCalculator, MetricsConfig
from agent_eval.tracer.schema import TokenCount, Trace, Turn


def make_trace(model: str, turns: list[Turn]) -> Trace:
    return Trace(model=model, turns=turns)


def make_turn(turn_id: int, prompt_tokens: int, completion_tokens: int) -> Turn:
    return Turn(
        turn_id=turn_id,
        role="assistant",
        content="",
        tokens=TokenCount(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.mark.asyncio
async def test_known_model_returns_correct_cost():
    # claude-sonnet-4-6: input=3.00/1M, output=15.00/1M
    # 1M input tokens + 1M output tokens = $3.00 + $15.00 = $18.00
    turns = [make_turn(0, 1_000_000, 1_000_000)]
    trace = make_trace("claude-sonnet-4-6", turns)
    result = await CostCalculator().analyze(trace)
    assert result.total_usd == pytest.approx(18.0)
    assert result.input_tokens == 1_000_000
    assert result.output_tokens == 1_000_000
    assert result.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_unknown_model_returns_zero_cost():
    turns = [make_turn(0, 500_000, 500_000)]
    trace = make_trace("unknown-model-xyz", turns)
    result = await CostCalculator().analyze(trace)
    assert result.total_usd == 0.0


@pytest.mark.asyncio
async def test_pricing_overrides_work():
    # Override claude-sonnet-4-6 to $1/1M input, $2/1M output
    config = MetricsConfig(
        pricing_overrides={"claude-sonnet-4-6": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
    )
    turns = [make_turn(0, 1_000_000, 1_000_000)]
    trace = make_trace("claude-sonnet-4-6", turns)
    result = await CostCalculator(config).analyze(trace)
    assert result.total_usd == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_cost_per_turn_computed_correctly():
    # claude-haiku-4-5: input=0.80/1M, output=4.00/1M
    turns = [
        make_turn(0, 100_000, 50_000),  # 0.1M * 0.80 + 0.05M * 4.00 = 0.08 + 0.20 = 0.28
        make_turn(1, 200_000, 100_000),  # 0.2M * 0.80 + 0.1M * 4.00  = 0.16 + 0.40 = 0.56
    ]
    trace = make_trace("claude-haiku-4-5", turns)
    result = await CostCalculator().analyze(trace)
    assert len(result.cost_per_turn) == 2
    assert result.cost_per_turn[0].turn_id == 0
    assert result.cost_per_turn[0].cost_usd == pytest.approx(0.28)
    assert result.cost_per_turn[1].turn_id == 1
    assert result.cost_per_turn[1].cost_usd == pytest.approx(0.56)
    assert result.total_usd == pytest.approx(0.84)


@pytest.mark.asyncio
async def test_empty_trace_returns_zero():
    trace = make_trace("gpt-4o", [])
    result = await CostCalculator().analyze(trace)
    assert result.total_usd == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_per_turn == []


@pytest.mark.asyncio
async def test_gpt4o_pricing():
    # gpt-4o: input=2.50/1M, output=10.00/1M
    turns = [make_turn(0, 2_000_000, 500_000)]
    trace = make_trace("gpt-4o", turns)
    result = await CostCalculator().analyze(trace)
    # 2M * 2.50 + 0.5M * 10.00 = 5.00 + 5.00 = 10.00
    assert result.total_usd == pytest.approx(10.0)
