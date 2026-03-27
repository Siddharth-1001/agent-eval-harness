"""End-to-end integration tests for the agent-eval CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run `agent-eval <args>` as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "agent_eval.cli", *args],
        capture_output=True,
        text=True,
    )


class TestVersionE2E:
    def test_version_exits_zero(self) -> None:
        result = _run("version")
        assert result.returncode == 0

    def test_version_outputs_version_string(self) -> None:
        from agent_eval import __version__

        result = _run("version")
        assert __version__ in result.stdout


class TestListE2E:
    def test_list_exits_zero_with_empty_dir(self, tmp_path: Path) -> None:
        result = _run("list", "--traces-dir", str(tmp_path))
        assert result.returncode == 0

    def test_list_exits_zero_with_nonexistent_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        result = _run("list", "--traces-dir", str(missing))
        assert result.returncode == 0


class TestShowE2E:
    def test_show_nonexistent_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run("show", "nonexistent-run-id", "--traces-dir", str(tmp_path))
        assert result.returncode != 0
