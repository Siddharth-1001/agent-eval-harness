"""Tests for the OpenAI Agents adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.adapters.openai_agents import trace_openai_agent
from agent_eval.tracer.writer import TraceWriterConfig


@pytest.fixture()
def writer_config(tmp_path: Path) -> TraceWriterConfig:
    return TraceWriterConfig(output_dir=tmp_path)


class TestTraceOpenaiAgent:
    def test_decorator_wraps_function(self, writer_config: TraceWriterConfig):
        """The decorator preserves the wrapped function's name."""

        @trace_openai_agent(task="t", model="gpt-4o", writer_config=writer_config)
        async def my_agent(prompt: str) -> str:
            return "hello"

        assert my_agent.__name__ == "my_agent"

    @pytest.mark.asyncio
    async def test_decorated_function_returns_result(self, writer_config: TraceWriterConfig):
        """The wrapped function still returns the original result."""

        @trace_openai_agent(task="t", model="gpt-4o", writer_config=writer_config)
        async def my_agent(prompt: str) -> str:
            return "agent response"

        result = await my_agent("hello")
        assert result == "agent response"

    @pytest.mark.asyncio
    async def test_trace_file_written(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """A trace file is written after the decorated function completes."""

        @trace_openai_agent(task="my-task", model="gpt-4o", writer_config=writer_config)
        async def my_agent(prompt: str) -> str:
            return "done"

        await my_agent("test input")

        trace_files = list(tmp_path.glob("*.json"))
        assert len(trace_files) == 1

    @pytest.mark.asyncio
    async def test_trace_contains_user_and_assistant_turns(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """The trace has a user turn (input) and an assistant turn (output)."""

        @trace_openai_agent(task="t", writer_config=writer_config)
        async def my_agent(prompt: str) -> str:
            return "the answer"

        await my_agent("the question")

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        roles = [t["role"] for t in data["turns"]]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_trace_captures_task_name(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """The task name specified in the decorator appears in the trace."""

        @trace_openai_agent(task="special-task", writer_config=writer_config)
        async def my_agent(prompt: str) -> str:
            return "ok"

        await my_agent("prompt")

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["task"] == "special-task"

    @pytest.mark.asyncio
    async def test_trace_written_even_on_exception(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """A trace is still written even when the wrapped function raises."""

        @trace_openai_agent(task="t", writer_config=writer_config)
        async def failing_agent(prompt: str) -> str:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            await failing_agent("prompt")

        # Trace file should still exist
        assert len(list(tmp_path.glob("*.json"))) == 1

    @pytest.mark.asyncio
    async def test_works_without_openai_agents_installed(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """Decorator works even when openai_agents package is not installed."""
        import sys
        from unittest.mock import patch

        with patch.dict("sys.modules", {"openai_agents": None}):
            # Remove cached adapter module to force re-import
            for key in list(sys.modules.keys()):
                if "agent_eval.adapters.openai_agents" in key:
                    del sys.modules[key]

            from agent_eval.adapters.openai_agents import trace_openai_agent as toa

            @toa(task="t", writer_config=writer_config)
            async def my_agent(prompt: str) -> str:
                return "result"

            result = await my_agent("input")
            assert result == "result"


class TestEvalRunHooks:
    """Tests for EvalRunHooks — the openai-agents RunHooks implementation."""

    @pytest.mark.asyncio
    async def test_on_tool_start_and_end_records_tool_call(self, writer_config):
        """on_tool_start + on_tool_end pair creates a tool turn with a ToolCall."""
        from agent_eval.adapters.openai_agents import EvalRunHooks
        from agent_eval.tracer.collector import TraceCollector

        collector = TraceCollector(model="gpt-4o", task="t")
        hooks = EvalRunHooks(collector)

        fake_tool = type("Tool", (), {"name": "search"})()
        await hooks.on_tool_start(None, None, fake_tool)
        await hooks.on_tool_end(None, None, fake_tool, "search results")

        trace = collector.finalize()
        tool_turns = [t for t in trace.turns if t.role == "tool"]
        assert len(tool_turns) == 1
        assert len(tool_turns[0].tool_calls) == 1
        assert tool_turns[0].tool_calls[0].tool_name == "search"
        assert tool_turns[0].tool_calls[0].output == "search results"
        assert tool_turns[0].tool_calls[0].success is True

    @pytest.mark.asyncio
    async def test_on_tool_end_without_start_is_noop(self):
        """Calling on_tool_end without a matching on_tool_start does not crash."""
        from agent_eval.adapters.openai_agents import EvalRunHooks
        from agent_eval.tracer.collector import TraceCollector

        collector = TraceCollector(model="gpt-4o")
        hooks = EvalRunHooks(collector)
        fake_tool = type("Tool", (), {"name": "tool"})()
        # Should not raise
        await hooks.on_tool_end(None, None, fake_tool, "result")

    @pytest.mark.asyncio
    async def test_multiple_sequential_tool_calls_are_all_recorded(self, writer_config):
        """Multiple sequential tool call pairs are each recorded correctly (FIFO)."""
        from agent_eval.adapters.openai_agents import EvalRunHooks
        from agent_eval.tracer.collector import TraceCollector

        collector = TraceCollector(model="gpt-4o")
        hooks = EvalRunHooks(collector)

        for name, result in [("search", "r1"), ("fetch", "r2"), ("store", "r3")]:
            fake_tool = type("T", (), {"name": name})()
            await hooks.on_tool_start(None, None, fake_tool)
            await hooks.on_tool_end(None, None, fake_tool, result)

        trace = collector.finalize()
        tool_calls = [tc for t in trace.turns for tc in t.tool_calls]
        assert len(tool_calls) == 3
        assert [tc.tool_name for tc in tool_calls] == ["search", "fetch", "store"]

    @pytest.mark.asyncio
    async def test_no_op_hooks_do_not_raise(self):
        """on_agent_start, on_agent_end, and on_handoff don't raise."""
        from agent_eval.adapters.openai_agents import EvalRunHooks
        from agent_eval.tracer.collector import TraceCollector

        hooks = EvalRunHooks(TraceCollector(model="m"))
        await hooks.on_agent_start(None, None)
        await hooks.on_agent_end(None, None, "output")
        await hooks.on_handoff(None, None, None)


class TestPatchRunnerOnce:
    def test_patch_runner_once_is_noop_when_package_missing(self):
        """_patch_runner_once does nothing (no error) when openai_agents isn't installed."""
        import sys
        from unittest.mock import patch

        from agent_eval.adapters import openai_agents as oa_module

        original = oa_module._RUNNER_PATCHED
        try:
            oa_module._RUNNER_PATCHED = False
            with patch.dict("sys.modules", {"openai_agents": None}):
                oa_module._patch_runner_once()  # must not raise
        finally:
            oa_module._RUNNER_PATCHED = original

    def test_patch_runner_once_called_twice_is_idempotent(self):
        """Calling _patch_runner_once twice doesn't re-patch or raise."""
        import sys
        from unittest.mock import patch

        from agent_eval.adapters import openai_agents as oa_module

        original = oa_module._RUNNER_PATCHED
        try:
            oa_module._RUNNER_PATCHED = True  # simulate already patched
            oa_module._patch_runner_once()  # should be a no-op
        finally:
            oa_module._RUNNER_PATCHED = original
