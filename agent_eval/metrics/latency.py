"""Latency analyzer."""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from agent_eval.tracer.schema import Trace


class ToolLatency(BaseModel):
    tool_name: str
    avg_latency_ms: float
    call_count: int


class LatencyMetrics(BaseModel):
    total_ms: int
    p50_ms: int
    p95_ms: int
    slowest_turn_id: int | None
    tool_latency_breakdown: list[ToolLatency]


class LatencyAnalyzer:
    """Stateless analyzer for latency metrics."""

    async def analyze(self, trace: Trace) -> LatencyMetrics:
        turn_latencies = [turn.latency_ms for turn in trace.turns]
        total_ms = sum(turn_latencies)

        if turn_latencies:
            sorted_latencies = sorted(turn_latencies)
            p50_ms = int(statistics.median(sorted_latencies))
            idx_p95 = max(0, int(len(sorted_latencies) * 0.95) - 1)
            p95_ms = sorted_latencies[idx_p95]
            slowest_turn_id = max(range(len(trace.turns)), key=lambda i: trace.turns[i].latency_ms)
        else:
            p50_ms = 0
            p95_ms = 0
            slowest_turn_id = None

        # Per-tool latency breakdown
        tool_data: dict[str, list[int]] = {}
        for turn in trace.turns:
            for call in turn.tool_calls:
                tool_data.setdefault(call.tool_name, []).append(call.latency_ms)

        tool_latency_breakdown = [
            ToolLatency(
                tool_name=name,
                avg_latency_ms=statistics.mean(latencies),
                call_count=len(latencies),
            )
            for name, latencies in tool_data.items()
        ]

        return LatencyMetrics(
            total_ms=total_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            slowest_turn_id=slowest_turn_id,
            tool_latency_breakdown=tool_latency_breakdown,
        )
