"""Metrics engine for agent-eval-harness."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from agent_eval.metrics.cost import CostCalculator, CostMetrics, MetricsConfig
from agent_eval.metrics.hallucination import (
    HallucinationConfig,
    HallucinationDetector,
    HallucinationMetrics,
    LLMJudge,
    ToolHallucinationConfig,
)
from agent_eval.metrics.latency import LatencyAnalyzer, LatencyMetrics
from agent_eval.metrics.tool_success import ToolMetrics, ToolSuccessAnalyzer
from agent_eval.tracer.schema import Trace


class MetricsReport(BaseModel):
    tool: ToolMetrics
    hallucination: HallucinationMetrics
    latency: LatencyMetrics
    cost: CostMetrics


async def compute_all_metrics(trace: Trace, config: MetricsConfig | None = None) -> MetricsReport:
    """Run all analyzers in parallel and return a full MetricsReport."""
    cfg = config or MetricsConfig()
    halluc_config = HallucinationConfig()

    async with asyncio.TaskGroup() as tg:
        tool_task = tg.create_task(ToolSuccessAnalyzer().analyze(trace))
        halluc_task = tg.create_task(HallucinationDetector(halluc_config).analyze(trace))
        latency_task = tg.create_task(LatencyAnalyzer().analyze(trace))
        cost_task = tg.create_task(CostCalculator(cfg).analyze(trace))

    return MetricsReport(
        tool=tool_task.result(),
        hallucination=halluc_task.result(),
        latency=latency_task.result(),
        cost=cost_task.result(),
    )


__all__ = [
    "MetricsReport",
    "MetricsConfig",
    "compute_all_metrics",
    "ToolSuccessAnalyzer",
    "ToolMetrics",
    "HallucinationDetector",
    "HallucinationConfig",
    "HallucinationMetrics",
    "ToolHallucinationConfig",
    "LLMJudge",
    "LatencyAnalyzer",
    "LatencyMetrics",
    "CostCalculator",
    "CostMetrics",
]
