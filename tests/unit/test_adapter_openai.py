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
