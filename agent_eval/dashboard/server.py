"""FastAPI dashboard server for agent-eval-harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

_STATIC_DIR = Path(__file__).parent / "static"


class CompareRequest(BaseModel):
    run_id_a: str
    run_id_b: str


def create_app(traces_dir: Path) -> FastAPI:
    """Create and return the FastAPI app bound to *traces_dir*."""

    app = FastAPI(title="agent-eval dashboard")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _all_trace_paths() -> list[Path]:
        if not traces_dir.exists():
            return []
        return sorted(
            traces_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _load_trace_json(run_id: str) -> dict[str, Any]:
        """Load raw trace dict or raise HTTPException(404)."""
        # Strip any path separators to prevent directory traversal attacks
        safe_id = Path(run_id).name
        paths = list(traces_dir.glob(f"{safe_id}*.json"))
        if not paths:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def _load_trace_obj(run_id: str):
        from agent_eval.tracer.schema import Trace

        raw = _load_trace_json(run_id)
        return Trace.model_validate(raw)

    # ── API routes ────────────────────────────────────────────────────────────

    @app.get("/api/runs")
    async def list_runs(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        from agent_eval.tracer.schema import Trace

        paths = _all_trace_paths()
        total = len(paths)
        start = (page - 1) * page_size
        end = start + page_size
        page_paths = paths[start:end]

        runs = []
        for p in page_paths:
            try:
                trace = Trace.model_validate_json(p.read_text(encoding="utf-8"))
                summary = trace.summary
                runs.append(
                    {
                        "run_id": trace.run_id,
                        "task": trace.task,
                        "model": trace.model,
                        "created_at": trace.created_at.isoformat(),
                        "turn_count": len(trace.turns),
                        "total_tool_calls": summary.total_tool_calls if summary else 0,
                        "tool_success_rate": summary.tool_success_rate if summary else None,
                        "estimated_cost_usd": summary.estimated_cost_usd if summary else None,
                    }
                )
            except Exception:
                pass

        return {"runs": runs, "total": total, "page": page, "page_size": page_size}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        return _load_trace_json(run_id)

    @app.get("/api/runs/{run_id}/metrics")
    async def get_run_metrics(run_id: str) -> dict[str, Any]:
        from agent_eval.metrics import compute_all_metrics

        trace = _load_trace_obj(run_id)
        report = await compute_all_metrics(trace)
        return report.model_dump()

    @app.post("/api/compare")
    async def compare_runs(body: CompareRequest) -> dict[str, Any]:
        from agent_eval.metrics import compute_all_metrics

        trace_a = _load_trace_obj(body.run_id_a)
        trace_b = _load_trace_obj(body.run_id_b)

        metrics_a, metrics_b = await asyncio.gather(
            compute_all_metrics(trace_a),
            compute_all_metrics(trace_b),
        )

        return {
            "run_id_a": body.run_id_a,
            "run_id_b": body.run_id_b,
            "metrics_a": metrics_a.model_dump(),
            "metrics_b": metrics_b.model_dump(),
        }

    # ── Static / SPA fallback ─────────────────────────────────────────────────

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> Any:
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            return JSONResponse(
                status_code=503,
                content={"detail": "Dashboard UI not available. Use /api/* endpoints."},
            )
        return FileResponse(str(index), media_type="text/html")

    return app
