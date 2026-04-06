"""CLI entrypoint for agent-eval-harness."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("agent_eval.cli")

app = typer.Typer(
    name="agent-eval",
    help="agent-eval-harness: evaluation framework for agentic AI systems.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)

DEFAULT_TRACES_DIR = Path("~/.agent-eval/traces/")

# Only allow UUIDs or valid prefixes (hex + hyphens) to prevent path injection
_RUN_ID_RE = re.compile(r"^[a-fA-F0-9\-]{1,36}$")


def _resolve_traces_dir(traces_dir: Path) -> Path:
    return traces_dir.expanduser().resolve()


def _load_trace(run_id: str, traces_dir: Path):
    """Load a single trace by run_id. Returns Trace or None."""
    from agent_eval.tracer.schema import Trace

    # Sanitize run_id to prevent path traversal
    if not _RUN_ID_RE.match(run_id):
        logger.warning("Invalid run_id format rejected: %s", run_id[:50])
        return None
    td = _resolve_traces_dir(traces_dir)
    safe_id = Path(run_id).name  # strip any remaining path components
    candidates = list(td.glob(f"{safe_id}*.json"))
    if not candidates:
        return None
    return Trace.model_validate_json(candidates[0].read_text(encoding="utf-8"))


def _list_traces(traces_dir: Path):
    """Return list of Trace objects from traces_dir, sorted by created_at desc."""
    from agent_eval.tracer.schema import Trace

    td = _resolve_traces_dir(traces_dir)
    if not td.exists():
        return []

    traces = []
    for f in sorted(td.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            traces.append(Trace.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Skipping invalid trace file: %s", f.name)
    return traces


@app.command("version")
def version() -> None:
    """Show the current version."""
    from agent_eval import __version__

    console.print(f"agent-eval-harness v{__version__}")


@app.command("run")
def run(
    task: Annotated[str, typer.Option("--task", help="Path to Python script to run")],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory to save trace output"),
    ] = DEFAULT_TRACES_DIR,
) -> None:
    """Run a Python script and save its trace."""
    task_path = Path(task).expanduser().resolve()
    if not task_path.exists():
        err_console.print(f"[red]Error:[/red] Task script not found: {task_path}")
        raise typer.Exit(1)

    output_dir = output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Running task:[/bold] {task_path}")
    console.print(f"[bold]Output dir:[/bold] {output_dir}")
    console.rule()

    env = {**__import__("os").environ, "AGENT_EVAL_TRACES_DIR": str(output_dir)}
    result = subprocess.run(
        [sys.executable, str(task_path)],
        env=env,
    )

    console.rule()
    if result.returncode == 0:
        console.print("[green]Task completed successfully.[/green]")
    else:
        console.print(f"[red]Task exited with code {result.returncode}.[/red]")
        raise typer.Exit(result.returncode)


@app.command("list")
def list_runs(
    traces_dir: Annotated[
        Path,
        typer.Option("--traces-dir", help="Directory containing trace files"),
    ] = DEFAULT_TRACES_DIR,
) -> None:
    """List all saved evaluation runs."""
    traces = _list_traces(traces_dir)

    table = Table(
        title="Evaluation Runs",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("run_id", style="dim", width=10)
    table.add_column("task")
    table.add_column("model")
    table.add_column("turns", justify="right")
    table.add_column("tool_calls", justify="right")
    table.add_column("success_rate", justify="right")
    table.add_column("cost_usd", justify="right")
    table.add_column("created_at")

    if not traces:
        table.add_row("-", "[dim]no runs found[/dim]", "-", "-", "-", "-", "-", "-")
    else:
        for trace in traces:
            summary = trace.summary
            run_id_short = trace.run_id[:8]
            task_name = trace.task or "-"
            model = trace.model or "-"
            turns = str(len(trace.turns))
            tool_calls = str(summary.total_tool_calls) if summary else "-"
            success_rate = f"{summary.tool_success_rate:.1%}" if summary else "-"
            cost = f"${summary.estimated_cost_usd:.4f}" if summary else "-"
            created = trace.created_at.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                run_id_short,
                task_name,
                model,
                turns,
                tool_calls,
                success_rate,
                cost,
                created,
            )

    console.print(table)


@app.command("show")
def show(
    run_id: Annotated[str, typer.Argument(help="Run ID (or prefix) to display")],
    traces_dir: Annotated[
        Path,
        typer.Option("--traces-dir", help="Directory containing trace files"),
    ] = DEFAULT_TRACES_DIR,
) -> None:
    """Show metrics for a single evaluation run."""
    trace = _load_trace(run_id, traces_dir)
    if trace is None:
        td = _resolve_traces_dir(traces_dir)
        err_console.print(f"[red]Error:[/red] Run '{run_id}' not found in {td}")
        raise typer.Exit(1)

    metrics = asyncio.run(_compute_metrics(trace))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # Tool metrics
    table.add_row("[cyan]Tool Success Rate[/cyan]", f"{metrics.tool.success_rate:.1%}")
    table.add_row("[cyan]Total Tool Calls[/cyan]", str(metrics.tool.total_calls))
    table.add_row("[cyan]Successful Calls[/cyan]", str(metrics.tool.successful_calls))
    table.add_row("[cyan]Failed Calls[/cyan]", str(metrics.tool.failed_calls))
    table.add_section()
    # Hallucination metrics
    halluc_rate = f"{metrics.hallucination.hallucination_rate:.1%}"
    table.add_row("[yellow]Hallucination Rate[/yellow]", halluc_rate)
    table.add_row("[yellow]Total Flags[/yellow]", str(metrics.hallucination.total_flags))
    table.add_section()
    # Latency metrics
    table.add_row("[green]Total Latency[/green]", f"{metrics.latency.total_ms} ms")
    table.add_row("[green]P50 Turn Latency[/green]", f"{metrics.latency.p50_ms} ms")
    table.add_row("[green]P95 Turn Latency[/green]", f"{metrics.latency.p95_ms} ms")
    table.add_section()
    # Cost metrics
    table.add_row("[red]Estimated Cost[/red]", f"${metrics.cost.total_usd:.6f}")
    table.add_row("[red]Input Tokens[/red]", str(metrics.cost.input_tokens))
    table.add_row("[red]Output Tokens[/red]", str(metrics.cost.output_tokens))

    panel = Panel(
        table,
        title=(
            f"[bold]Run: {trace.run_id[:8]}[/bold]"
            f"  |  {trace.task or 'unknown task'}  |  {trace.model}"
        ),
        subtitle=f"Created: {trace.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    console.print(panel)


@app.command("compare")
def compare(
    run_id_a: Annotated[str, typer.Argument(help="First run ID")],
    run_id_b: Annotated[str, typer.Argument(help="Second run ID")],
    traces_dir: Annotated[
        Path,
        typer.Option("--traces-dir", help="Directory containing trace files"),
    ] = DEFAULT_TRACES_DIR,
    export: Annotated[
        Path | None,
        typer.Option("--export", help="Export comparison to HTML file"),
    ] = None,
) -> None:
    """Compare metrics between two evaluation runs."""
    trace_a = _load_trace(run_id_a, traces_dir)
    trace_b = _load_trace(run_id_b, traces_dir)

    if trace_a is None:
        err_console.print(f"[red]Error:[/red] Run '{run_id_a}' not found.")
        raise typer.Exit(1)
    if trace_b is None:
        err_console.print(f"[red]Error:[/red] Run '{run_id_b}' not found.")
        raise typer.Exit(1)

    metrics_a, metrics_b = asyncio.run(_compute_both_metrics(trace_a, trace_b))

    rows = _build_comparison_rows(metrics_a, metrics_b)

    table = Table(
        title="Run Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column(f"A: {run_id_a[:8]}", justify="right")
    table.add_column(f"B: {run_id_b[:8]}", justify="right")
    table.add_column("Delta", justify="right")

    for metric_name, val_a, val_b, delta_str in rows:
        table.add_row(metric_name, val_a, val_b, delta_str)

    console.print(table)

    if export is not None:
        html = _render_compare_html(
            run_id_a,
            run_id_b,
            trace_a,
            trace_b,
            metrics_a,
            metrics_b,
            rows,
        )
        export_path = Path(export).expanduser().resolve()
        export_path.write_text(html, encoding="utf-8")
        console.print(f"[green]Exported comparison to:[/green] {export_path}")


@app.command("dashboard")
def dashboard(
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 7000,
    traces_dir: Annotated[
        Path,
        typer.Option("--traces-dir", help="Directory containing trace files"),
    ] = DEFAULT_TRACES_DIR,
) -> None:
    """Start the local evaluation dashboard."""
    import uvicorn

    from agent_eval.dashboard.server import create_app

    td = _resolve_traces_dir(traces_dir)
    td.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Starting dashboard[/bold green] on http://127.0.0.1:{port}")
    console.print(f"Traces directory: {td}")
    console.print("Press Ctrl+C to stop.")

    dash_app = create_app(td)
    uvicorn.run(dash_app, host="127.0.0.1", port=port)


# ── helpers ──────────────────────────────────────────────────────────────────


async def _compute_metrics(trace):
    from agent_eval.metrics import compute_all_metrics

    return await compute_all_metrics(trace)


async def _compute_both_metrics(trace_a, trace_b):
    from agent_eval.metrics import compute_all_metrics

    return await asyncio.gather(
        compute_all_metrics(trace_a),
        compute_all_metrics(trace_b),
    )


def _fmt_delta(a: float, b: float, higher_is_better: bool = True, pct: bool = False) -> str:
    delta = b - a
    delta_str = f"{delta:+.1%}" if pct else f"{delta:+.4f}"
    if delta == 0:
        return delta_str
    good = (delta > 0) == higher_is_better
    arrow = "▲" if delta > 0 else "▼"
    color = "green" if good else "red"
    return f"[{color}]{arrow} {delta_str}[/{color}]"


def _build_comparison_rows(ma, mb) -> list[tuple[str, str, str, str]]:
    rows = []
    rows.append(
        (
            "Tool Success Rate",
            f"{ma.tool.success_rate:.1%}",
            f"{mb.tool.success_rate:.1%}",
            _fmt_delta(ma.tool.success_rate, mb.tool.success_rate, higher_is_better=True, pct=True),
        )
    )
    rows.append(
        (
            "Total Tool Calls",
            str(ma.tool.total_calls),
            str(mb.tool.total_calls),
            _fmt_delta(ma.tool.total_calls, mb.tool.total_calls, higher_is_better=True),
        )
    )
    rows.append(
        (
            "Hallucination Rate",
            f"{ma.hallucination.hallucination_rate:.1%}",
            f"{mb.hallucination.hallucination_rate:.1%}",
            _fmt_delta(
                ma.hallucination.hallucination_rate,
                mb.hallucination.hallucination_rate,
                higher_is_better=False,
                pct=True,
            ),
        )
    )
    rows.append(
        (
            "Total Latency (ms)",
            str(ma.latency.total_ms),
            str(mb.latency.total_ms),
            _fmt_delta(ma.latency.total_ms, mb.latency.total_ms, higher_is_better=False),
        )
    )
    rows.append(
        (
            "Estimated Cost (USD)",
            f"${ma.cost.total_usd:.6f}",
            f"${mb.cost.total_usd:.6f}",
            _fmt_delta(ma.cost.total_usd, mb.cost.total_usd, higher_is_better=False),
        )
    )
    return rows


def _render_compare_html(run_id_a, run_id_b, trace_a, trace_b, ma, mb, rows) -> str:
    def strip_markup(s: str) -> str:
        import re

        return re.sub(r"\[/?[^\]]+\]", "", s)

    # Escape all user-controlled strings before embedding in HTML
    e_id_a = html.escape(run_id_a)
    e_id_a8 = html.escape(run_id_a[:8])
    e_id_b = html.escape(run_id_b)
    e_id_b8 = html.escape(run_id_b[:8])
    e_task_a = html.escape(trace_a.task or "unknown")
    e_model_a = html.escape(trace_a.model or "unknown")
    e_task_b = html.escape(trace_b.task or "unknown")
    e_model_b = html.escape(trace_b.model or "unknown")

    rows_html = ""
    for metric_name, val_a, val_b, delta_str in rows:
        delta_clean = strip_markup(delta_str)
        if "green" in delta_str:
            delta_color = "#4ade80"
        elif "red" in delta_str:
            delta_color = "#f87171"
        else:
            delta_color = "#e5e7eb"
        rows_html += (
            f"<tr>"
            f"<td>{metric_name}</td>"
            f"<td>{val_a}</td>"
            f"<td>{val_b}</td>"
            f"<td style='color:{delta_color}'>{delta_clean}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>agent-eval comparison: {e_id_a8} vs {e_id_b8}</title>
  <style>
    body {{ font-family: monospace; background: #111827; color: #e5e7eb; margin: 2rem; }}
    h1 {{ color: #60a5fa; }}
    .meta {{ color: #9ca3af; margin-bottom: 1.5rem; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th {{ background: #1e3a5f; color: #93c5fd; padding: 0.6rem 1rem; text-align: left; }}
    td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #374151; }}
    tr:hover {{ background: #1f2937; }}
    .run-a {{ color: #a78bfa; }}
    .run-b {{ color: #34d399; }}
  </style>
</head>
<body>
  <h1>agent-eval: Run Comparison</h1>
  <div class="meta">
    <span class="run-a">A: {e_id_a} — {e_task_a} ({e_model_a})</span><br/>
    <span class="run-b">B: {e_id_b} — {e_task_b} ({e_model_b})</span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th class="run-a">A: {e_id_a8}</th>
        <th class="run-b">B: {e_id_b8}</th>
        <th>Delta (B - A)</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</body>
</html>
"""


if __name__ == "__main__":
    app()
