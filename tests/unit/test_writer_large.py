"""Tests for TraceWriter large-trace truncation path."""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.tracer.schema import Trace, Turn
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


def _make_big_trace(num_turns: int = 5, content_len: int = 200_000) -> Trace:
    turns = []
    for i in range(num_turns):
        turns.append(Turn(turn_id=i, role="assistant", content="z" * content_len))
    return Trace(model="big-model", turns=turns)


def test_write_large_trace_is_truncated(tmp_path: Path) -> None:
    """A trace exceeding max_trace_size_mb triggers extra turn-content truncation."""
    config = TraceWriterConfig(output_dir=tmp_path, max_trace_size_mb=0.001)
    writer = TraceWriter(config)
    trace = _make_big_trace(num_turns=2, content_len=10_000)
    path = writer.write(trace)
    assert path.exists()
    data = json.loads(path.read_text())
    # Content should have been trimmed to 1000 chars + truncation marker
    for turn in data["turns"]:
        assert len(turn["content"]) <= 1000 + 100  # marker overhead


async def test_async_write_large_trace_is_truncated(tmp_path: Path) -> None:
    """Async writer also truncates oversized traces."""
    config = TraceWriterConfig(output_dir=tmp_path, max_trace_size_mb=0.001)
    writer = TraceWriter(config)
    trace = _make_big_trace(num_turns=2, content_len=10_000)
    path = await writer.async_write(trace)
    assert path.exists()
    data = json.loads(path.read_text())
    for turn in data["turns"]:
        assert len(turn["content"]) <= 1000 + 100
