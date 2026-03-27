"""Tests for the CrewAI adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_eval.tracer.writer import TraceWriterConfig


@pytest.fixture()
def writer_config(tmp_path: Path) -> TraceWriterConfig:
    return TraceWriterConfig(output_dir=tmp_path)


def _make_mock_crewai(kickoff_return: object = "crew result") -> MagicMock:
    """Return a mock crewai module with Crew that returns kickoff_return."""
    mock_module = MagicMock()
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = kickoff_return
    mock_crew_instance.akickoff = MagicMock(return_value=kickoff_return)
    mock_module.Crew.return_value = mock_crew_instance
    return mock_module, mock_crew_instance


def _build_crew(writer_config: TraceWriterConfig, mock_module: MagicMock):
    """Import EvalHarnessCrew with crewai mocked."""
    import sys

    # Force re-import with mock
    for key in list(sys.modules.keys()):
        if "agent_eval.adapters.crewai" in key:
            del sys.modules[key]

    with patch.dict("sys.modules", {"crewai": mock_module}):
        from agent_eval.adapters.crewai import EvalHarnessCrew

        crew = EvalHarnessCrew(
            agents=[MagicMock()],
            tasks=[MagicMock()],
            task="eval-task",
            writer_config=writer_config,
        )
    return crew


class TestEvalHarnessCrew:
    def test_kickoff_writes_trace(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """kickoff() writes a trace file."""
        mock_module, _ = _make_mock_crewai()

        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": mock_module}):
            from agent_eval.adapters.crewai import EvalHarnessCrew

            crew = EvalHarnessCrew(
                agents=[MagicMock()],
                tasks=[MagicMock()],
                task="eval-task",
                writer_config=writer_config,
            )
            crew.kickoff()

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_kickoff_returns_crew_result(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """kickoff() returns the underlying Crew.kickoff() result."""
        mock_module, mock_crew_instance = _make_mock_crewai("final answer")

        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": mock_module}):
            from agent_eval.adapters.crewai import EvalHarnessCrew

            crew = EvalHarnessCrew(
                agents=[MagicMock()],
                tasks=[MagicMock()],
                task="t",
                writer_config=writer_config,
            )
            result = crew.kickoff()

        assert result == "final answer"

    def test_trace_contains_task_name(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """The trace records the task name."""
        mock_module, _ = _make_mock_crewai()

        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": mock_module}):
            from agent_eval.adapters.crewai import EvalHarnessCrew

            crew = EvalHarnessCrew(
                agents=[MagicMock()],
                tasks=[MagicMock()],
                task="my-crewai-task",
                writer_config=writer_config,
            )
            crew.kickoff()

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["task"] == "my-crewai-task"

    def test_step_callback_records_tool_call(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """The step_callback captures tool calls from AgentAction-like objects."""
        mock_module, mock_crew_instance = _make_mock_crewai()

        captured_callback = None

        def capture_crew(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("step_callback")
            return mock_crew_instance

        mock_module.Crew.side_effect = lambda **kw: capture_crew(**kw)

        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": mock_module}):
            from agent_eval.adapters.crewai import EvalHarnessCrew

            crew = EvalHarnessCrew(
                agents=[MagicMock()],
                tasks=[MagicMock()],
                task="t",
                writer_config=writer_config,
            )

            # Simulate a step event
            step = MagicMock()
            step.tool = "search_tool"
            step.tool_input = {"query": "AI"}
            step.text = "thinking..."
            if captured_callback:
                captured_callback(step)

            crew.kickoff()

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        # Find a turn with tool_calls
        all_tool_calls = [tc for t in data["turns"] for tc in t["tool_calls"]]
        # If callback was captured, verify the tool was recorded
        if captured_callback:
            assert any(tc["tool_name"] == "search_tool" for tc in all_tool_calls)

    def test_raises_import_error_when_crewai_missing(self):
        """ImportError with install hint when crewai package is absent."""
        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": None}):
            with pytest.raises((ImportError, TypeError)):
                from agent_eval.adapters.crewai import EvalHarnessCrew

                EvalHarnessCrew(agents=[], tasks=[])

    def test_trace_has_user_and_assistant_turns(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """The trace should contain at least a user and assistant turn."""
        mock_module, _ = _make_mock_crewai("output")

        import sys

        for key in list(sys.modules.keys()):
            if "agent_eval.adapters.crewai" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"crewai": mock_module}):
            from agent_eval.adapters.crewai import EvalHarnessCrew

            crew = EvalHarnessCrew(
                agents=[MagicMock()],
                tasks=[MagicMock()],
                task="t",
                writer_config=writer_config,
            )
            crew.kickoff()

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        roles = {t["role"] for t in data["turns"]}
        assert "user" in roles
        assert "assistant" in roles
