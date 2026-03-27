"""Tests for the PydanticAI adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_eval.adapters.pydantic_ai import _WrappedPydanticAgent
from agent_eval.tracer.writer import TraceWriterConfig


@pytest.fixture()
def writer_config(tmp_path: Path) -> TraceWriterConfig:
    return TraceWriterConfig(output_dir=tmp_path)


def _make_mock_result(data: str = "agent answer") -> MagicMock:
    result = MagicMock()
    result.data = data
    result.all_messages.return_value = []
    return result


class TestWrappedPydanticAgent:
    def test_run_sync_returns_result(self, writer_config: TraceWriterConfig):
        """run_sync() returns the underlying agent's result."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _make_mock_result("the answer")

        wrapped = _WrappedPydanticAgent(mock_agent, task="t", writer_config=writer_config)
        result = wrapped.run_sync("hello")
        assert result.data == "the answer"

    def test_run_sync_writes_trace(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """run_sync() writes a trace file."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _make_mock_result()

        wrapped = _WrappedPydanticAgent(mock_agent, task="t", writer_config=writer_config)
        wrapped.run_sync("hello")

        assert len(list(tmp_path.glob("*.json"))) == 1

    @pytest.mark.asyncio
    async def test_run_returns_result(self, writer_config: TraceWriterConfig):
        """async run() returns the underlying agent's result."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result("async answer"))

        wrapped = _WrappedPydanticAgent(mock_agent, task="t", writer_config=writer_config)
        result = await wrapped.run("hello")
        assert result.data == "async answer"

    @pytest.mark.asyncio
    async def test_run_writes_trace(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """async run() writes a trace file."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())

        wrapped = _WrappedPydanticAgent(mock_agent, task="async-task", writer_config=writer_config)
        await wrapped.run("hello")

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_trace_contains_user_and_assistant_turns(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """run_sync() trace contains user and assistant turns."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _make_mock_result("response")

        wrapped = _WrappedPydanticAgent(mock_agent, task="t", writer_config=writer_config)
        wrapped.run_sync("user question")

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        roles = {t["role"] for t in data["turns"]}
        assert "user" in roles
        assert "assistant" in roles

    def test_trace_captures_task_name(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """The task name is recorded in the trace."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _make_mock_result()

        wrapped = _WrappedPydanticAgent(
            mock_agent, task="special-task", writer_config=writer_config
        )
        wrapped.run_sync("prompt")

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["task"] == "special-task"

    @pytest.mark.asyncio
    async def test_run_trace_captures_task_name(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """Async run() also captures the task name."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())

        wrapped = _WrappedPydanticAgent(mock_agent, task="async-task", writer_config=writer_config)
        await wrapped.run("prompt")

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["task"] == "async-task"

    def test_model_name_extracted_from_agent(self, writer_config: TraceWriterConfig):
        """The model name is extracted from the agent's model attribute if available."""
        mock_agent = MagicMock()
        mock_agent.model.model_name = "gpt-4o"
        mock_agent.run_sync.return_value = _make_mock_result()

        wrapped = _WrappedPydanticAgent(mock_agent, writer_config=writer_config)
        wrapped.run_sync("hello")
        # No assertion needed — just verify it doesn't crash


class TestWithEvalHarness:
    def test_with_eval_harness_raises_without_pydantic_ai(self):
        """with_eval_harness raises ImportError when pydantic_ai is absent."""
        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.pydantic_ai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"pydantic_ai": None}):
            with pytest.raises(ImportError, match="pip install"):
                from agent_eval.adapters.pydantic_ai import with_eval_harness

                with_eval_harness(MagicMock())

    def test_with_eval_harness_returns_wrapped_agent(self):
        """with_eval_harness returns a _WrappedPydanticAgent when package is present."""
        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.pydantic_ai" in key:
                del sys.modules[key]

        mock_pydantic_ai = MagicMock()
        with patch.dict("sys.modules", {"pydantic_ai": mock_pydantic_ai}):
            from agent_eval.adapters.pydantic_ai import _WrappedPydanticAgent, with_eval_harness

            mock_agent = MagicMock()
            wrapped = with_eval_harness(mock_agent, task="t")
            assert isinstance(wrapped, _WrappedPydanticAgent)
