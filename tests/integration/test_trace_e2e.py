"""Integration tests: end-to-end trace creation and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval import AgentTracer, TraceCollector, TraceWriter, TraceWriterConfig, trace_agent
from agent_eval.tracer.schema import TokenCount, ToolCall, Trace

# ---------------------------------------------------------------------------
# @trace_agent decorator — sync
# ---------------------------------------------------------------------------


def test_trace_agent_sync_creates_file(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="sync_test", model="gpt-4o", writer_config=config)
    def my_agent(question: str) -> str:
        return f"Answer to: {question}"

    result = my_agent("What is 2+2?")
    assert result == "Answer to: What is 2+2?"

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


def test_trace_agent_sync_file_content_valid(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="validate_content", model="claude-sonnet-4-6", writer_config=config)
    def my_agent(q: str) -> str:
        return "response"

    my_agent("input")
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1

    data = json.loads(json_files[0].read_text())
    trace = Trace.model_validate(data)
    assert trace.model == "claude-sonnet-4-6"
    assert trace.task == "validate_content"
    assert len(trace.turns) >= 1


def test_trace_agent_sync_records_user_turn(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="turn_test", model="gpt-4o", writer_config=config)
    def my_agent(q: str) -> str:
        return "ok"

    my_agent("my question")
    json_files = list(tmp_path.glob("*.json"))
    trace = Trace.model_validate_json(json_files[0].read_text())
    roles = [t.role for t in trace.turns]
    assert "user" in roles
    assert "assistant" in roles


def test_trace_agent_sync_preserves_return_value(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="return_value", model="gpt-4o", writer_config=config)
    def compute(x: int) -> int:
        return x * 2

    result = compute(21)
    assert result == 42


def test_trace_agent_sync_exception_still_writes_trace(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="error_test", model="gpt-4o", writer_config=config)
    def failing_agent(q: str) -> str:
        raise ValueError("intentional error")

    with pytest.raises(ValueError, match="intentional error"):
        failing_agent("question")

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


# ---------------------------------------------------------------------------
# @trace_agent decorator — async
# ---------------------------------------------------------------------------


async def test_trace_agent_async_creates_file(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="async_test", model="gpt-4o", writer_config=config)
    async def async_agent(question: str) -> str:
        return f"Async answer: {question}"

    result = await async_agent("hello async")
    assert result == "Async answer: hello async"

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


async def test_trace_agent_async_file_content_valid(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="async_validate", model="claude-sonnet-4-6", writer_config=config)
    async def async_agent(q: str) -> str:
        return "async response"

    await async_agent("async input")
    json_files = list(tmp_path.glob("*.json"))
    trace = Trace.model_validate_json(json_files[0].read_text())
    assert trace.model == "claude-sonnet-4-6"
    assert trace.task == "async_validate"


async def test_trace_agent_async_preserves_return_value(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="async_return", model="gpt-4o", writer_config=config)
    async def double(x: int) -> int:
        return x * 2

    result = await double(10)
    assert result == 20


async def test_trace_agent_async_exception_still_writes_trace(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="async_error", model="gpt-4o", writer_config=config)
    async def failing_async(q: str) -> str:
        raise RuntimeError("async error")

    with pytest.raises(RuntimeError, match="async error"):
        await failing_async("question")

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


# ---------------------------------------------------------------------------
# AgentTracer context manager — sync
# ---------------------------------------------------------------------------


def test_agent_tracer_sync_creates_file(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    with AgentTracer(task="cm_sync", model="gpt-4o", writer_config=config) as tracer:
        turn_id = tracer.collector.start_turn("user", "question")
        tracer.collector.end_turn(turn_id, latency_ms=100)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


def test_agent_tracer_sync_trace_accessible(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    with AgentTracer(task="cm_trace_access", model="gpt-4o", writer_config=config) as tracer:
        turn_id = tracer.collector.start_turn("user", "hi")
        tracer.collector.end_turn(turn_id, latency_ms=50)

    assert tracer.trace is not None
    assert isinstance(tracer.trace, Trace)
    assert tracer.trace.task == "cm_trace_access"


def test_agent_tracer_sync_file_content_valid(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    with AgentTracer(task="cm_valid", model="claude-sonnet-4-6", writer_config=config) as tracer:
        turn_id = tracer.collector.start_turn("user", "test message")
        call = ToolCall(
            tool_name="calculator",
            input_args={"expr": "2+2"},
            output="4",
            success=True,
            latency_ms=5,
        )
        tracer.collector.record_tool_call(turn_id, call)
        tracer.collector.end_turn(turn_id, latency_ms=30)

    json_files = list(tmp_path.glob("*.json"))
    trace = Trace.model_validate_json(json_files[0].read_text())
    assert trace.model == "claude-sonnet-4-6"
    assert len(trace.turns) == 1
    assert len(trace.turns[0].tool_calls) == 1
    assert trace.turns[0].tool_calls[0].tool_name == "calculator"


# ---------------------------------------------------------------------------
# AgentTracer context manager — async
# ---------------------------------------------------------------------------


async def test_agent_tracer_async_creates_file(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    async with AgentTracer(task="cm_async", model="gpt-4o", writer_config=config) as tracer:
        turn_id = await tracer.collector.async_start_turn("user", "async question")
        await tracer.collector.async_end_turn(turn_id, latency_ms=80)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


async def test_agent_tracer_async_file_content_valid(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    async with AgentTracer(
        task="cm_async_valid", model="claude-sonnet-4-6", writer_config=config
    ) as tracer:
        turn_id = await tracer.collector.async_start_turn("user", "async message")
        call = ToolCall(
            tool_name="web_search",
            input_args={"query": "test"},
            output="search results",
            success=True,
            latency_ms=200,
        )
        await tracer.collector.async_record_tool_call(turn_id, call)
        await tracer.collector.async_end_turn(
            turn_id,
            latency_ms=250,
            tokens=TokenCount(prompt_tokens=100, completion_tokens=60),
        )

    json_files = list(tmp_path.glob("*.json"))
    trace = Trace.model_validate_json(json_files[0].read_text())
    assert trace.model == "claude-sonnet-4-6"
    assert trace.task == "cm_async_valid"
    assert len(trace.turns) == 1
    assert trace.turns[0].tokens.total == 160


async def test_agent_tracer_async_trace_accessible(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    async with AgentTracer(task="cm_async_access", model="gpt-4o", writer_config=config) as tracer:
        turn_id = await tracer.collector.async_start_turn("user", "msg")
        await tracer.collector.async_end_turn(turn_id, latency_ms=10)

    assert tracer.trace is not None
    assert isinstance(tracer.trace, Trace)


# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------


def test_trace_file_is_valid_json(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)

    @trace_agent(task="json_valid", model="gpt-4o", writer_config=config)
    def agent(q: str) -> str:
        return "answer"

    agent("question")
    json_files = list(tmp_path.glob("*.json"))
    raw = json_files[0].read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert "run_id" in parsed
    assert "schema_version" in parsed
    assert "turns" in parsed
    assert "model" in parsed


def test_trace_file_schema_version_is_string_one(tmp_path: Path) -> None:
    config = TraceWriterConfig(output_dir=tmp_path)
    writer = TraceWriter(config)
    collector = TraceCollector(model="gpt-4o")
    trace = collector.finalize()
    path = writer.write(trace)
    data = json.loads(path.read_text())
    assert data["schema_version"] == "1"
