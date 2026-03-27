"""Unit tests for agent_eval.tracer.writer."""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.tracer.schema import ToolCall, Trace, Turn
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


def make_trace_with_long_content(content_len: int = 100, output_len: int = 100) -> Trace:
    """Helper to build a trace with specified content/output lengths."""
    call = ToolCall(
        tool_name="tool",
        input_args={},
        output="x" * output_len,
        success=True,
        latency_ms=10,
    )
    turn = Turn(
        turn_id=0,
        role="assistant",
        content="a" * content_len,
        tool_calls=[call],
    )
    return Trace(model="gpt-4o", turns=[turn])


class TestTraceWriterSync:
    def test_write_creates_file(self, tmp_path: Path, sample_trace: Trace) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = writer.write(sample_trace)
        assert path.exists()
        assert path.suffix == ".json"

    def test_write_filename_matches_run_id(self, tmp_path: Path, sample_trace: Trace) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = writer.write(sample_trace)
        assert path.name == f"{sample_trace.run_id}.json"

    def test_written_json_validates_back_to_trace(
        self, tmp_path: Path, sample_trace: Trace
    ) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = writer.write(sample_trace)
        json_str = path.read_text(encoding="utf-8")
        restored = Trace.model_validate_json(json_str)
        assert restored.run_id == sample_trace.run_id
        assert restored.model == sample_trace.model

    def test_output_dir_created_if_not_exists(self, tmp_path: Path, sample_trace: Trace) -> None:
        nested = tmp_path / "a" / "b" / "c"
        config = TraceWriterConfig(output_dir=nested)
        writer = TraceWriter(config)
        path = writer.write(sample_trace)
        assert path.exists()

    def test_truncation_tool_output(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(output_dir=tmp_path, max_output_chars=10)
        writer = TraceWriter(config)
        trace = make_trace_with_long_content(output_len=200)
        path = writer.write(trace)
        data = json.loads(path.read_text())
        output = data["turns"][0]["tool_calls"][0]["output"]
        assert len(output) > 10  # includes truncation marker
        assert "[truncated" in output
        assert output.startswith("x" * 10)

    def test_truncation_turn_content(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(output_dir=tmp_path, max_content_chars=20)
        writer = TraceWriter(config)
        trace = make_trace_with_long_content(content_len=500)
        path = writer.write(trace)
        data = json.loads(path.read_text())
        content = data["turns"][0]["content"]
        assert "[truncated" in content
        assert content.startswith("a" * 20)

    def test_no_truncation_within_limits(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(
            output_dir=tmp_path, max_output_chars=1000, max_content_chars=1000
        )
        writer = TraceWriter(config)
        trace = make_trace_with_long_content(content_len=50, output_len=50)
        path = writer.write(trace)
        data = json.loads(path.read_text())
        assert data["turns"][0]["content"] == "a" * 50
        assert data["turns"][0]["tool_calls"][0]["output"] == "x" * 50

    def test_truncation_marker_contains_original_length(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(output_dir=tmp_path, max_output_chars=5)
        writer = TraceWriter(config)
        trace = make_trace_with_long_content(output_len=100)
        path = writer.write(trace)
        data = json.loads(path.read_text())
        output = data["turns"][0]["tool_calls"][0]["output"]
        assert "100" in output

    def test_default_config(self, sample_trace: Trace) -> None:
        writer = TraceWriter()
        assert writer.config.max_output_chars == 10_000
        assert writer.config.max_content_chars == 50_000

    def test_write_returns_path_object(self, tmp_path: Path, sample_trace: Trace) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        result = writer.write(sample_trace)
        assert isinstance(result, Path)

    def test_write_trace_without_turns(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        trace = Trace(model="gpt-4o")
        path = writer.write(trace)
        assert path.exists()
        restored = Trace.model_validate_json(path.read_text())
        assert restored.turns == []


class TestTraceWriterAsync:
    async def test_async_write_creates_file(self, tmp_path: Path, sample_trace: Trace) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = await writer.async_write(sample_trace)
        assert path.exists()

    async def test_async_write_filename_matches_run_id(
        self, tmp_path: Path, sample_trace: Trace
    ) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = await writer.async_write(sample_trace)
        assert path.name == f"{sample_trace.run_id}.json"

    async def test_async_written_json_validates_back_to_trace(
        self, tmp_path: Path, sample_trace: Trace
    ) -> None:
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        path = await writer.async_write(sample_trace)
        json_str = path.read_text(encoding="utf-8")
        restored = Trace.model_validate_json(json_str)
        assert restored.run_id == sample_trace.run_id
        assert restored.model == sample_trace.model

    async def test_async_write_truncation(self, tmp_path: Path) -> None:
        config = TraceWriterConfig(output_dir=tmp_path, max_output_chars=5)
        writer = TraceWriter(config)
        trace = make_trace_with_long_content(output_len=100)
        path = await writer.async_write(trace)
        data = json.loads(path.read_text())
        output = data["turns"][0]["tool_calls"][0]["output"]
        assert "[truncated" in output
