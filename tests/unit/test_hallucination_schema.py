"""Tests for hallucination detection in schema mode."""

from __future__ import annotations

import pytest

from agent_eval.metrics.hallucination import (
    HallucinationConfig,
    HallucinationDetector,
    ToolHallucinationConfig,
    _schema_check,
)
from agent_eval.tracer.schema import ToolCall, Trace, Turn


def make_call(tool_name: str, input_args: dict) -> ToolCall:
    return ToolCall(tool_name=tool_name, input_args=input_args, success=True, latency_ms=10)


def make_trace_with_call(
    tool_name: str, input_args: dict, tool_config: ToolHallucinationConfig
) -> Trace:
    call = make_call(tool_name, input_args)
    turn = Turn(turn_id=0, role="assistant", content="test content", tool_calls=[call])
    config = HallucinationConfig(tools={tool_name: tool_config})
    return Trace(model="claude-sonnet-4-6", turns=[turn]), config, call


def test_missing_required_field_is_flagged():
    schema = {
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }
    call = make_call("search", {"query": "hello"})  # missing 'limit'
    flags = _schema_check(call, schema)
    flag_args = [f.argument_name for f in flags]
    assert "limit" in flag_args
    assert all(f.method == "schema" for f in flags)
    assert all(f.confidence == 1.0 for f in flags)


def test_wrong_type_is_flagged():
    schema = {
        "required": [],
        "properties": {
            "count": {"type": "integer"},
        },
    }
    call = make_call("tool", {"count": "not-an-int"})
    flags = _schema_check(call, schema)
    assert len(flags) == 1
    assert flags[0].argument_name == "count"
    assert "integer" in flags[0].expected
    assert flags[0].received == "not-an-int"


def test_valid_call_produces_no_flags():
    schema = {
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }
    call = make_call("search", {"query": "hello", "limit": 10})
    flags = _schema_check(call, schema)
    assert flags == []


def test_unknown_type_does_not_flag():
    schema = {
        "required": [],
        "properties": {
            "data": {"type": "custom_type"},
        },
    }
    call = make_call("tool", {"data": "anything"})
    flags = _schema_check(call, schema)
    assert flags == []


def test_no_schema_returns_no_flags():
    call = make_call("tool", {"arg": "value"})
    flags = _schema_check(call, None)
    assert flags == []


@pytest.mark.asyncio
async def test_detector_schema_mode_flags_missing_required():
    schema = {
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }
    tool_config = ToolHallucinationConfig(mode="schema", json_schema=schema)
    call = make_call("read_file", {})  # missing 'path'
    turn = Turn(turn_id=0, role="assistant", content="content", tool_calls=[call])
    config = HallucinationConfig(tools={"read_file": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 1
    assert metrics.flags_by_tool.get("read_file", 0) == 1
    assert call.hallucination_flags[0].argument_name == "path"


@pytest.mark.asyncio
async def test_detector_schema_mode_no_flags_on_valid_call():
    schema = {
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }
    tool_config = ToolHallucinationConfig(mode="schema", json_schema=schema)
    call = make_call("read_file", {"path": "/tmp/file.txt"})
    turn = Turn(turn_id=0, role="assistant", content="content", tool_calls=[call])
    config = HallucinationConfig(tools={"read_file": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 0
    assert call.hallucination_flags == []


def test_boolean_is_not_accepted_as_integer():
    """True/False must not pass an 'integer' type check (bool is a subclass of int)."""
    schema = {
        "required": [],
        "properties": {
            "count": {"type": "integer"},
        },
    }
    call = make_call("tool", {"count": True})
    flags = _schema_check(call, schema)
    assert len(flags) == 1
    assert flags[0].argument_name == "count"


def test_boolean_is_accepted_for_boolean_type():
    """True/False must still pass a 'boolean' type check."""
    schema = {
        "required": [],
        "properties": {
            "enabled": {"type": "boolean"},
        },
    }
    call = make_call("tool", {"enabled": True})
    flags = _schema_check(call, schema)
    assert flags == []


def test_integer_is_not_accepted_for_boolean_type():
    """An integer (1) must not satisfy a 'boolean' type constraint."""
    schema = {
        "required": [],
        "properties": {
            "enabled": {"type": "boolean"},
        },
    }
    call = make_call("tool", {"enabled": 1})
    flags = _schema_check(call, schema)
    assert len(flags) == 1
