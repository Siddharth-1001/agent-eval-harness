"""Tests for the LangChain/LangGraph adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_eval.adapters.langchain import LangGraphTracer, _EvalCallbackHandler, _make_handler
from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.writer import TraceWriterConfig


@pytest.fixture()
def writer_config(tmp_path: Path) -> TraceWriterConfig:
    return TraceWriterConfig(output_dir=tmp_path)


@pytest.fixture()
def collector() -> TraceCollector:
    return TraceCollector(model="test-model", task="test-task")


@pytest.fixture()
def handler(collector: TraceCollector) -> _EvalCallbackHandler:
    return _EvalCallbackHandler(collector)


class TestEvalCallbackHandler:
    def test_on_tool_start_records_input(self, handler: _EvalCallbackHandler):
        """on_tool_start stores timing and input for the run."""
        handler.on_tool_start({}, '{"query": "hello"}', run_id="run-1")
        assert "run-1" in handler._tool_start_times
        assert handler._tool_inputs["run-1"] == {"query": "hello"}

    def test_on_tool_start_non_json_input(self, handler: _EvalCallbackHandler):
        """Non-JSON input_str is stored as {'input': value}."""
        handler.on_tool_start({}, "some plain text", run_id="run-2")
        assert handler._tool_inputs["run-2"] == {"input": "some plain text"}

    def test_on_tool_end_creates_tool_call(
        self, handler: _EvalCallbackHandler, collector: TraceCollector
    ):
        """on_tool_end produces a ToolCall attached to the collector."""
        handler.on_tool_start({}, '{"city": "Paris"}', run_id="r1")
        # Create an assistant turn so tool call has a home
        turn_id = collector.start_turn("assistant", "checking…")
        handler._current_turn_id = turn_id

        handler.on_tool_end("sunny", run_id="r1", name="get_weather")
        collector.end_turn(turn_id)

        trace = collector.finalize()
        assert len(trace.turns) == 1
        assert len(trace.turns[0].tool_calls) == 1
        tc = trace.turns[0].tool_calls[0]
        assert tc.tool_name == "get_weather"
        assert tc.input_args == {"city": "Paris"}
        assert tc.output == "sunny"
        assert tc.success is True

    def test_on_tool_end_without_prior_turn_creates_tool_turn(
        self, handler: _EvalCallbackHandler, collector: TraceCollector
    ):
        """When there's no current_turn_id, on_tool_end still records a turn."""
        handler.on_tool_start({}, "{}", run_id="r2")
        handler.on_tool_end("result", run_id="r2", name="my_tool")

        trace = collector.finalize()
        assert any(t.role == "tool" for t in trace.turns)

    def test_on_tool_error_creates_failed_tool_call(
        self, handler: _EvalCallbackHandler, collector: TraceCollector
    ):
        """on_tool_error records a failed ToolCall."""
        handler.on_tool_start({}, "{}", run_id="r3")
        turn_id = collector.start_turn("assistant", "")
        handler._current_turn_id = turn_id

        handler.on_tool_error(ValueError("boom"), run_id="r3")
        collector.end_turn(turn_id)

        trace = collector.finalize()
        tc = trace.turns[0].tool_calls[0]
        assert tc.success is False

    def test_on_llm_end_records_assistant_turn(
        self, handler: _EvalCallbackHandler, collector: TraceCollector
    ):
        """on_llm_end records an assistant turn from generation text."""
        mock_gen = MagicMock()
        mock_gen.text = "Hello there"
        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)

        trace = collector.finalize()
        assert any(t.role == "assistant" and "Hello there" in t.content for t in trace.turns)


class TestLangGraphTracer:
    @pytest.mark.asyncio
    async def test_async_context_manager_writes_trace(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """Async context manager exit writes a trace file."""
        async with LangGraphTracer(task="t1", model="gpt-4", writer_config=writer_config):
            pass

        trace_files = list(tmp_path.glob("*.json"))
        assert len(trace_files) == 1

    def test_sync_context_manager_writes_trace(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """Sync context manager exit writes a trace file."""
        with LangGraphTracer(task="t2", model="gpt-4", writer_config=writer_config):
            pass

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_langgraph_config_contains_callbacks(self, writer_config: TraceWriterConfig):
        """langgraph_config exposes the callbacks key."""
        tracer = LangGraphTracer(task="t3", writer_config=writer_config)
        cfg = tracer.langgraph_config
        assert "callbacks" in cfg
        assert len(cfg["callbacks"]) == 1

    @pytest.mark.asyncio
    async def test_tool_calls_captured_in_trace(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """Tool start/end events during a run are captured in the trace."""
        async with LangGraphTracer(task="t4", writer_config=writer_config) as tracer:
            handler = tracer._handler
            handler.on_tool_start({}, '{"q": "test"}', run_id="run-a")

            # Simulate an assistant turn existing
            turn_id = tracer._collector.start_turn("assistant", "running")
            handler._current_turn_id = turn_id
            handler.on_tool_end("done", run_id="run-a", name="my_tool")
            tracer._collector.end_turn(turn_id)

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assistant_turns = [t for t in data["turns"] if t["role"] == "assistant"]
        assert any(len(t["tool_calls"]) > 0 for t in assistant_turns)

    def test_make_handler_falls_back_without_langchain(self, collector: TraceCollector):
        """_make_handler returns an _EvalCallbackHandler even without langchain installed."""
        import sys

        modules_backup = {}
        for key in list(sys.modules.keys()):
            if "langchain" in key:
                modules_backup[key] = sys.modules.pop(key)

        try:
            from unittest.mock import patch

            with patch.dict(
                "sys.modules", {"langchain_core": None, "langchain_core.callbacks": None}
            ):
                handler = _make_handler(collector)
                assert isinstance(handler, _EvalCallbackHandler)
        finally:
            sys.modules.update(modules_backup)
