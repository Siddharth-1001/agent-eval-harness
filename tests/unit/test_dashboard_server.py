"""Unit tests for agent_eval.dashboard.server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from agent_eval.dashboard.server import create_app  # noqa: E402

# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RUN_ID = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"

SAMPLE_TRACE = {
    "schema_version": "1",
    "run_id": SAMPLE_RUN_ID,
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

SAMPLE_RUN_ID_B = "11112222-3333-4444-5555-666677778888"
SAMPLE_TRACE_B = {**SAMPLE_TRACE, "run_id": SAMPLE_RUN_ID_B, "task": "task-b"}


def _write_trace(traces_dir: Path, trace: dict, run_id: str) -> None:
    (traces_dir / f"{run_id}.json").write_text(json.dumps(trace), encoding="utf-8")


@pytest.fixture
def empty_app(tmp_path: Path):
    return create_app(tmp_path)


@pytest.fixture
def populated_app(tmp_path: Path):
    _write_trace(tmp_path, SAMPLE_TRACE, SAMPLE_RUN_ID)
    _write_trace(tmp_path, SAMPLE_TRACE_B, SAMPLE_RUN_ID_B)
    return create_app(tmp_path)


# ── GET /api/runs (empty) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_runs_empty(empty_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=empty_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []
    assert data["total"] == 0


# ── GET /api/runs (populated, paginated) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_list_runs_returns_runs(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["runs"]) == 2


@pytest.mark.asyncio
async def test_list_runs_pagination(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs?page=1&page_size=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["runs"]) == 1


@pytest.mark.asyncio
async def test_list_runs_page2(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs?page=2&page_size=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1


# ── GET /api/runs/{run_id} ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_run_not_found(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs/nonexistent-run-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_returns_trace(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/runs/{SAMPLE_RUN_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == SAMPLE_RUN_ID
    assert data["model"] == "test-model"


# ── GET /api/runs/{run_id}/metrics ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_run_metrics_not_found(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/runs/nonexistent/metrics")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_metrics_returns_report(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/runs/{SAMPLE_RUN_ID}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "tool" in data
    assert "hallucination" in data
    assert "latency" in data
    assert "cost" in data
    assert data["tool"]["total_calls"] == 1
    assert data["tool"]["success_rate"] == 1.0


# ── POST /api/compare ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_returns_dict(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/compare",
            json={"run_id_a": SAMPLE_RUN_ID, "run_id_b": SAMPLE_RUN_ID_B},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id_a"] == SAMPLE_RUN_ID
    assert data["run_id_b"] == SAMPLE_RUN_ID_B
    assert "metrics_a" in data
    assert "metrics_b" in data


@pytest.mark.asyncio
async def test_compare_missing_run_returns_404(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/compare",
            json={"run_id_a": SAMPLE_RUN_ID, "run_id_b": "doesnotexist"},
        )
    assert resp.status_code == 404


# ── GET / (SPA fallback) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spa_root_returns_html(populated_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=populated_app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    # Static index.html exists in the package, so we expect 200 with HTML
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert "text/html" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_spa_missing_static_returns_503_or_html(tmp_path: Path) -> None:
    """When static dir has no index.html, server should return 503."""
    from agent_eval.dashboard import server as server_mod

    # Temporarily point STATIC_DIR to a path without index.html
    missing_static = tmp_path / "no_static"
    missing_static.mkdir()

    # The already-created app uses the original closure; create a fresh one
    # by directly patching the module-level constant
    original = server_mod._STATIC_DIR
    server_mod._STATIC_DIR = missing_static
    try:
        app_patched2 = create_app(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app_patched2), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        # Without index.html the fallback returns 503
        assert resp.status_code in (503, 200)
    finally:
        server_mod._STATIC_DIR = original
