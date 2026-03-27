"""Unit tests for agent_eval.cli."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_eval import __version__
from agent_eval.cli import app

runner = CliRunner()


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_trace_file(tmp_path: Path, run_id: str = "aaaabbbb-0000-0000-0000-000000000000") -> Path:
    """Write a minimal valid trace JSON to tmp_path and return the file path."""
    trace = {
        "schema_version": "1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00+00:00",
        "model": "test-model",
        "task": "test-task",
        "agent_config": {},
        "turns": [
            {
                "turn_id": 0,
                "role": "assistant",
                "content": "hello",
                "tool_calls": [
                    {
                        "call_id": "c1",
                        "tool_name": "search_web",
                        "input_args": {"query": "foo"},
                        "output": "bar",
                        "success": True,
                        "latency_ms": 100,
                        "hallucination_flags": [],
                    }
                ],
                "latency_ms": 200,
                "tokens": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ],
        "summary": None,
    }
    p = tmp_path / f"{run_id}.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    return p


# ── version ───────────────────────────────────────────────────────────────────


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_command_contains_package_name() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "agent-eval-harness" in result.output


# ── list ──────────────────────────────────────────────────────────────────────


def test_list_no_traces_dir_handles_gracefully(tmp_path: Path) -> None:
    """list command with a nonexistent traces dir should exit 0 and not crash."""
    missing = tmp_path / "no_such_dir"
    result = runner.invoke(app, ["list", "--traces-dir", str(missing)])
    assert result.exit_code == 0


def test_list_empty_dir_shows_table(tmp_path: Path) -> None:
    result = runner.invoke(app, ["list", "--traces-dir", str(tmp_path)])
    assert result.exit_code == 0
    # Table header columns present
    assert "run_id" in result.output or "Run" in result.output


def test_list_with_trace_shows_run(tmp_path: Path) -> None:
    _make_trace_file(tmp_path)
    result = runner.invoke(app, ["list", "--traces-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "aaaabbbb" in result.output  # truncated run_id


# ── show ──────────────────────────────────────────────────────────────────────


def test_show_missing_run_id_shows_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["show", "nonexistent", "--traces-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_show_existing_run_displays_metrics(tmp_path: Path) -> None:
    run_id = "aaaabbbb-0000-0000-0000-000000000000"
    _make_trace_file(tmp_path, run_id)
    result = runner.invoke(app, ["show", run_id, "--traces-dir", str(tmp_path)])
    assert result.exit_code == 0
    # Panel content includes metric labels
    assert "Tool Success Rate" in result.output or "success" in result.output.lower()


def test_show_prefix_match(tmp_path: Path) -> None:
    run_id = "ccccdddd-1111-1111-1111-111111111111"
    _make_trace_file(tmp_path, run_id)
    result = runner.invoke(app, ["show", "ccccdddd", "--traces-dir", str(tmp_path)])
    assert result.exit_code == 0


# ── compare ───────────────────────────────────────────────────────────────────


def test_compare_matching_run_ids_shows_table(tmp_path: Path) -> None:
    run_id_a = "aaaabbbb-0000-0000-0000-000000000000"
    run_id_b = "ccccdddd-1111-1111-1111-111111111111"
    _make_trace_file(tmp_path, run_id_a)
    _make_trace_file(tmp_path, run_id_b)

    result = runner.invoke(
        app,
        ["compare", run_id_a, run_id_b, "--traces-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    # Should show a comparison table with both run IDs truncated
    assert "aaaabbbb" in result.output
    assert "ccccdddd" in result.output


def test_compare_missing_first_run_shows_error(tmp_path: Path) -> None:
    run_id_b = "ccccdddd-1111-1111-1111-111111111111"
    _make_trace_file(tmp_path, run_id_b)
    result = runner.invoke(
        app,
        ["compare", "nonexistent", run_id_b, "--traces-dir", str(tmp_path)],
    )
    assert result.exit_code != 0


def test_compare_export_creates_html(tmp_path: Path) -> None:
    run_id_a = "aaaabbbb-0000-0000-0000-000000000000"
    run_id_b = "ccccdddd-1111-1111-1111-111111111111"
    _make_trace_file(tmp_path, run_id_a)
    _make_trace_file(tmp_path, run_id_b)
    export_path = tmp_path / "compare.html"

    result = runner.invoke(
        app,
        [
            "compare",
            run_id_a,
            run_id_b,
            "--traces-dir",
            str(tmp_path),
            "--export",
            str(export_path),
        ],
    )
    assert result.exit_code == 0
    assert export_path.exists()
    html = export_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "aaaabbbb" in html


# ── help ──────────────────────────────────────────────────────────────────────


def test_help_flag_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_help_shows_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "list", "show", "compare", "dashboard", "version"):
        assert cmd in result.output
