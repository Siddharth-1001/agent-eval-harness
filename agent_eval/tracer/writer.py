from __future__ import annotations

import functools
import logging
import statistics
import tomllib
from pathlib import Path

from pydantic import BaseModel

from agent_eval.tracer.schema import RunSummary, Trace

logger = logging.getLogger("agent_eval.writer")


@functools.cache
def _load_pricing() -> dict:
    """Load pricing.toml once and cache it for the process lifetime."""
    pricing_path = Path(__file__).parent.parent / "data" / "pricing.toml"
    try:
        with open(pricing_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        logger.warning("Could not load pricing.toml; cost estimates will be zero")
        return {}


def _compute_run_summary(trace: Trace) -> RunSummary:
    """
    Compute a :class:`RunSummary` synchronously from a trace.

    Covers tool counts, latency percentiles, and cost estimate (via pricing.toml).
    ``hallucination_rate`` is left as ``0.0`` — compute it on demand with
    :class:`~agent_eval.metrics.hallucination.HallucinationDetector`.
    """
    # Tool metrics
    all_calls = [tc for turn in trace.turns for tc in turn.tool_calls]
    total_tc = len(all_calls)
    successful_tc = sum(1 for tc in all_calls if tc.success)
    tool_success_rate = successful_tc / total_tc if total_tc > 0 else 0.0

    # Latency
    turn_latencies = [t.latency_ms for t in trace.turns]
    total_ms = sum(turn_latencies)
    if turn_latencies:
        sorted_lat = sorted(turn_latencies)
        p50_ms = int(statistics.median(sorted_lat))
        idx95 = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95_ms = sorted_lat[idx95]
    else:
        p50_ms = p95_ms = 0

    # Cost estimate (mirrors CostCalculator logic; avoids async overhead at write time)
    cost_usd = 0.0
    try:
        _pricing = _load_pricing()
        _model_cfg = _pricing.get("models", {}).get(trace.model, {})
        _input_rate = _model_cfg.get("input_per_1m", 0.0)
        _output_rate = _model_cfg.get("output_per_1m", 0.0)
        _total_input = sum(t.tokens.prompt_tokens for t in trace.turns)
        _total_output = sum(t.tokens.completion_tokens for t in trace.turns)
        cost_usd = (_total_input * _input_rate + _total_output * _output_rate) / 1_000_000
    except Exception:
        logger.debug("Cost estimation failed for model %s", trace.model)

    return RunSummary(
        turn_count=len(trace.turns),
        total_tool_calls=total_tc,
        successful_tool_calls=successful_tc,
        failed_tool_calls=total_tc - successful_tc,
        tool_success_rate=tool_success_rate,
        hallucination_rate=0.0,
        total_latency_ms=total_ms,
        p50_turn_latency_ms=p50_ms,
        p95_turn_latency_ms=p95_ms,
        estimated_cost_usd=cost_usd,
    )


class TraceWriterConfig(BaseModel):
    output_dir: Path = Path("~/.agent-eval/traces/")
    max_output_chars: int = 10_000
    max_content_chars: int = 50_000
    max_trace_size_mb: float = 5.0
    truncation_marker: str = "... [truncated, {original_len} chars total]"

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: object) -> None:
        self.output_dir = self.output_dir.expanduser()


class TraceWriter:
    """Writes Trace objects to JSON files with optional truncation."""

    def __init__(self, config: TraceWriterConfig | None = None) -> None:
        self.config = config or TraceWriterConfig()

    def _truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        marker = self.config.truncation_marker.format(original_len=len(text))
        return text[:max_chars] + marker

    def _apply_truncation(self, trace: Trace) -> Trace:
        """Return a copy of the trace with truncation applied."""
        data = trace.model_dump()
        for turn in data["turns"]:
            if isinstance(turn["content"], str):
                turn["content"] = self._truncate(turn["content"], self.config.max_content_chars)
            for call in turn["tool_calls"]:
                if isinstance(call["output"], str):
                    call["output"] = self._truncate(call["output"], self.config.max_output_chars)
        return Trace.model_validate(data)

    def write(self, trace: Trace) -> Path:
        """Write trace to disk synchronously. Returns the file path."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        # Compute summary and attach before truncation (truncation only trims text,
        # not metrics data, so computing from the original trace is correct).
        trace_with_summary = trace.model_copy(update={"summary": _compute_run_summary(trace)})
        truncated = self._apply_truncation(trace_with_summary)
        output_path = self.config.output_dir / f"{trace.run_id}.json"
        json_str = truncated.model_dump_json(indent=2)
        # Check size limit
        size_mb = len(json_str.encode()) / (1024 * 1024)
        if size_mb > self.config.max_trace_size_mb:
            # Truncate oldest turns' content
            data = truncated.model_dump()
            for turn in data["turns"]:
                turn["content"] = self._truncate(turn["content"], 1000)
            truncated = Trace.model_validate(data)
            json_str = truncated.model_dump_json(indent=2)
        output_path.write_text(json_str, encoding="utf-8")
        return output_path

    async def async_write(self, trace: Trace) -> Path:
        """Write trace to disk asynchronously."""
        import aiofiles

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        trace_with_summary = trace.model_copy(update={"summary": _compute_run_summary(trace)})
        truncated = self._apply_truncation(trace_with_summary)
        output_path = self.config.output_dir / f"{trace.run_id}.json"
        json_str = truncated.model_dump_json(indent=2)
        size_mb = len(json_str.encode()) / (1024 * 1024)
        if size_mb > self.config.max_trace_size_mb:
            data = truncated.model_dump()
            for turn in data["turns"]:
                turn["content"] = self._truncate(turn["content"], 1000)
            truncated = Trace.model_validate(data)
            json_str = truncated.model_dump_json(indent=2)
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json_str)
        return output_path
