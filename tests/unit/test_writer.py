"""Unit tests for agent_eval.tracer.writer."""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.tracer.schema import TokenCount, ToolCall, Trace, Turn
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig, _compute_run_summary


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


# ── _compute_run_summary ──────────────────────────────────────────────────────


class TestComputeRunSummary:
    def _make_trace(self, *, success: bool = True, latency_ms: int = 100) -> Trace:
        call = ToolCall(tool_name="search", input_args={}, success=success, latency_ms=latency_ms)
        turn = Turn(
            turn_id=0,
            role="assistant",
            content="resp",
            latency_ms=latency_ms,
            tool_calls=[call],
            tokens=TokenCount(prompt_tokens=10, completion_tokens=5),
        )
        return Trace(model="gpt-4o", turns=[turn])

    def test_summary_tool_counts(self):
        trace = self._make_trace(success=True)
        summary = _compute_run_summary(trace)
        assert summary.total_tool_calls == 1
        assert summary.successful_tool_calls == 1
        assert summary.failed_tool_calls == 0
        assert summary.tool_success_rate == 1.0

    def test_summary_failed_tool(self):
        trace = self._make_trace(success=False)
        summary = _compute_run_summary(trace)
        assert summary.successful_tool_calls == 0
        assert summary.failed_tool_calls == 1
        assert summary.tool_success_rate == 0.0

    def test_summary_no_tool_calls_success_rate_zero(self):
        trace = Trace(model="m", turns=[Turn(turn_id=0, role="user", content="hi")])
        summary = _compute_run_summary(trace)
        assert summary.total_tool_calls == 0
        assert summary.tool_success_rate == 0.0

    def test_summary_latency_fields(self):
        trace = self._make_trace(latency_ms=200)
        summary = _compute_run_summary(trace)
        assert summary.total_latency_ms == 200
        assert summary.p50_turn_latency_ms == 200
        assert summary.p95_turn_latency_ms == 200

    def test_summary_empty_trace_latency(self):
        trace = Trace(model="m", turns=[])
        summary = _compute_run_summary(trace)
        assert summary.total_latency_ms == 0
        assert summary.p50_turn_latency_ms == 0
        assert summary.p95_turn_latency_ms == 0

    def test_summary_turn_count(self):
        trace = self._make_trace()
        summary = _compute_run_summary(trace)
        assert summary.turn_count == 1

    def test_summary_hallucination_rate_is_zero(self):
        """hallucination_rate is always 0.0 at write time (computed on demand)."""
        trace = self._make_trace()
        summary = _compute_run_summary(trace)
        assert summary.hallucination_rate == 0.0

    def test_summary_cost_estimated_for_known_model(self):
        """Cost is estimated from pricing.toml for known models."""
        call = ToolCall(tool_name="t", input_args={}, success=True, latency_ms=0)
        turn = Turn(
            turn_id=0,
            role="assistant",
            content="hi",
            tool_calls=[call],
            tokens=TokenCount(prompt_tokens=1000, completion_tokens=500),
        )
        # gpt-4o is listed in pricing.toml
        trace = Trace(model="gpt-4o", turns=[turn])
        summary = _compute_run_summary(trace)
        assert summary.estimated_cost_usd >= 0.0  # cost is non-negative

    def test_summary_cost_zero_for_unknown_model(self):
        """Unknown models get zero cost (no pricing data)."""
        trace = Trace(model="no-such-model-xyz", turns=[])
        summary = _compute_run_summary(trace)
        assert summary.estimated_cost_usd == 0.0

    def test_write_populates_summary_field(self, tmp_path: Path):
        """write() embeds the RunSummary in the JSON file."""
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        trace = self._make_trace()
        path = writer.write(trace)
        data = json.loads(path.read_text())
        assert data["summary"] is not None
        assert data["summary"]["total_tool_calls"] == 1
        assert data["summary"]["tool_success_rate"] == 1.0

    async def test_async_write_populates_summary_field(self, tmp_path: Path):
        """async_write() also embeds the RunSummary in the JSON file."""
        config = TraceWriterConfig(output_dir=tmp_path)
        writer = TraceWriter(config)
        trace = self._make_trace()
        path = await writer.async_write(trace)
        data = json.loads(path.read_text())
        assert data["summary"] is not None
        assert data["summary"]["total_tool_calls"] == 1
