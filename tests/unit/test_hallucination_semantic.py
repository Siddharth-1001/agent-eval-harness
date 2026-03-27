"""Tests for hallucination detection in semantic mode."""

from __future__ import annotations

import pytest

from agent_eval.metrics.hallucination import (
    HallucinationConfig,
    HallucinationDetector,
    ToolHallucinationConfig,
    _semantic_check,
)
from agent_eval.tracer.schema import ToolCall, Trace, Turn


def make_call(tool_name: str, input_args: dict) -> ToolCall:
    return ToolCall(tool_name=tool_name, input_args=input_args, success=True, latency_ms=10)


def test_value_not_in_allowed_set_is_flagged():
    value_sets = {"status": ["active", "inactive", "pending"]}
    call = make_call("update_user", {"status": "deleted"})
    flags = _semantic_check(call, value_sets)
    assert len(flags) == 1
    assert flags[0].argument_name == "status"
    assert flags[0].received == "deleted"
    assert flags[0].method == "semantic"
    assert flags[0].confidence == pytest.approx(0.9)


def test_value_in_allowed_set_produces_no_flag():
    value_sets = {"status": ["active", "inactive", "pending"]}
    call = make_call("update_user", {"status": "active"})
    flags = _semantic_check(call, value_sets)
    assert flags == []


def test_missing_arg_not_in_input_args_is_skipped():
    value_sets = {"priority": ["low", "medium", "high"]}
    call = make_call("create_task", {"title": "Do something"})  # 'priority' not in input_args
    flags = _semantic_check(call, value_sets)
    assert flags == []


def test_multiple_value_sets_flags_all_violations():
    value_sets = {
        "status": ["open", "closed"],
        "priority": ["low", "medium", "high"],
    }
    call = make_call("update", {"status": "unknown", "priority": "critical"})
    flags = _semantic_check(call, value_sets)
    assert len(flags) == 2
    flag_args = {f.argument_name for f in flags}
    assert "status" in flag_args
    assert "priority" in flag_args


@pytest.mark.asyncio
async def test_detector_semantic_mode_flags_invalid_value():
    value_sets = {"direction": ["north", "south", "east", "west"]}
    tool_config = ToolHallucinationConfig(mode="semantic", value_sets=value_sets)
    call = make_call("navigate", {"direction": "up"})
    turn = Turn(turn_id=0, role="assistant", content="navigate up", tool_calls=[call])
    config = HallucinationConfig(tools={"navigate": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 1
    assert "semantic" in metrics.flags_by_method
    assert call.hallucination_flags[0].argument_name == "direction"


@pytest.mark.asyncio
async def test_detector_semantic_mode_with_schema_and_value_sets():
    schema = {
        "required": ["color", "size"],
        "properties": {
            "color": {"type": "string"},
            "size": {"type": "string"},
        },
    }
    value_sets = {"color": ["red", "green", "blue"]}
    tool_config = ToolHallucinationConfig(mode="semantic", schema=schema, value_sets=value_sets)
    # Missing 'size' (schema violation) + invalid color (semantic violation)
    call = make_call("paint", {"color": "purple"})
    turn = Turn(turn_id=0, role="assistant", content="paint", tool_calls=[call])
    config = HallucinationConfig(tools={"paint": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 2
    methods = {f.method for f in call.hallucination_flags}
    assert "schema" in methods
    assert "semantic" in methods
