"""Unit tests for agent_eval.tracer.collector."""

from __future__ import annotations

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import TokenCount, ToolCall, Trace


class TestTraceCollectorSync:
    def test_start_turn_returns_zero_for_first_turn(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("user", "hello")
        assert turn_id == 0

    def test_start_turn_increments(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        id0 = collector.start_turn("user", "first")
        id1 = collector.start_turn("assistant", "second")
        id2 = collector.start_turn("user", "third")
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

    def test_record_tool_call_appends_to_correct_turn(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        collector.start_turn("user", "do something")
        turn_id = collector.start_turn("assistant", "calling tool")
        call = ToolCall(
            tool_name="my_tool",
            input_args={"x": 1},
            output="result",
            success=True,
            latency_ms=50,
        )
        collector.record_tool_call(turn_id, call)
        assert len(collector._turns[turn_id].tool_calls) == 1
        assert collector._turns[0].tool_calls == []

    def test_record_multiple_tool_calls(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("assistant", "calling tools")
        for i in range(3):
            call = ToolCall(
                tool_name=f"tool_{i}",
                input_args={},
                output=None,
                success=True,
                latency_ms=10,
            )
            collector.record_tool_call(turn_id, call)
        assert len(collector._turns[turn_id].tool_calls) == 3

    def test_end_turn_sets_explicit_latency(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("user", "hello")
        collector.end_turn(turn_id, latency_ms=123)
        assert collector._turns[turn_id].latency_ms == 123

    def test_end_turn_auto_latency_is_positive(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("user", "hello")
        collector.end_turn(turn_id)
        assert collector._turns[turn_id].latency_ms >= 0

    def test_end_turn_sets_tokens(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("assistant", "response")
        tokens = TokenCount(prompt_tokens=100, completion_tokens=50)
        collector.end_turn(turn_id, tokens=tokens)
        assert collector._turns[turn_id].tokens.prompt_tokens == 100
        assert collector._turns[turn_id].tokens.completion_tokens == 50

    def test_finalize_returns_trace(self) -> None:
        collector = TraceCollector(model="gpt-4o", task="test task")
        turn_id = collector.start_turn("user", "question")
        collector.end_turn(turn_id, latency_ms=50)
        trace = collector.finalize()
        assert isinstance(trace, Trace)
        assert trace.model == "gpt-4o"
        assert trace.task == "test task"

    def test_finalize_preserves_turns(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        collector.start_turn("user", "msg1")
        collector.start_turn("assistant", "msg2")
        trace = collector.finalize()
        assert len(trace.turns) == 2
        assert trace.turns[0].content == "msg1"
        assert trace.turns[1].content == "msg2"

    def test_finalize_preserves_agent_config(self) -> None:
        config = {"temperature": 0.5, "max_tokens": 1000}
        collector = TraceCollector(model="gpt-4o", agent_config=config)
        trace = collector.finalize()
        assert trace.agent_config == config

    def test_agent_config_defaults_empty(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        trace = collector.finalize()
        assert trace.agent_config == {}

    def test_task_none_by_default(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        trace = collector.finalize()
        assert trace.task is None

    def test_end_turn_without_start_time_graceful(self) -> None:
        """If turn_start_times entry is missing, latency stays 0."""
        collector = TraceCollector(model="gpt-4o")
        turn_id = collector.start_turn("user", "hello")
        # Remove start time to simulate edge case
        del collector._turn_start_times[turn_id]
        collector.end_turn(turn_id)
        assert collector._turns[turn_id].latency_ms == 0


class TestTraceCollectorAsync:
    async def test_async_start_turn_returns_zero(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = await collector.async_start_turn("user", "hello")
        assert turn_id == 0

    async def test_async_start_turn_increments(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        id0 = await collector.async_start_turn("user", "first")
        id1 = await collector.async_start_turn("assistant", "second")
        assert id0 == 0
        assert id1 == 1

    async def test_async_record_tool_call(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = await collector.async_start_turn("assistant", "calling")
        call = ToolCall(
            tool_name="tool",
            input_args={"a": 1},
            output="out",
            success=True,
            latency_ms=20,
        )
        await collector.async_record_tool_call(turn_id, call)
        assert len(collector._turns[turn_id].tool_calls) == 1

    async def test_async_end_turn_sets_latency(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = await collector.async_start_turn("user", "hello")
        await collector.async_end_turn(turn_id, latency_ms=200)
        assert collector._turns[turn_id].latency_ms == 200

    async def test_async_end_turn_sets_tokens(self) -> None:
        collector = TraceCollector(model="gpt-4o")
        turn_id = await collector.async_start_turn("assistant", "response")
        tokens = TokenCount(prompt_tokens=80, completion_tokens=40)
        await collector.async_end_turn(turn_id, tokens=tokens)
        assert collector._turns[turn_id].tokens.total == 120

    async def test_async_finalize_returns_trace(self) -> None:
        collector = TraceCollector(model="claude-sonnet-4-6", task="async test")
        turn_id = await collector.async_start_turn("user", "question")
        await collector.async_end_turn(turn_id, latency_ms=100)
        trace = await collector.async_finalize()
        assert isinstance(trace, Trace)
        assert trace.model == "claude-sonnet-4-6"
        assert trace.task == "async test"
        assert len(trace.turns) == 1

    async def test_async_full_workflow(self) -> None:
        collector = TraceCollector(model="gpt-4o", task="full workflow")
        turn_id = await collector.async_start_turn("user", "do something")
        call = ToolCall(
            tool_name="web_search",
            input_args={"query": "test"},
            output="results",
            success=True,
            latency_ms=150,
        )
        await collector.async_record_tool_call(turn_id, call)
        await collector.async_end_turn(
            turn_id,
            latency_ms=200,
            tokens=TokenCount(prompt_tokens=50, completion_tokens=30),
        )
        trace = await collector.async_finalize()
        assert len(trace.turns) == 1
        assert len(trace.turns[0].tool_calls) == 1
        assert trace.turns[0].latency_ms == 200
        assert trace.turns[0].tokens.total == 80
