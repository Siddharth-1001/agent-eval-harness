"""Tool success rate analyzer."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from agent_eval.tracer.schema import Trace


class ToolBreakdown(BaseModel):
    tool_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float


class ToolMetrics(BaseModel):
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    per_tool: list[ToolBreakdown]


class ToolSuccessAnalyzer:
    """Stateless analyzer for tool call success/failure metrics."""

    async def analyze(self, trace: Trace) -> ToolMetrics:
        per_tool: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "failed": 0}
        )
        total = 0
        successful = 0

        for turn in trace.turns:
            for call in turn.tool_calls:
                total += 1
                per_tool[call.tool_name]["total"] += 1
                if call.success:
                    successful += 1
                    per_tool[call.tool_name]["success"] += 1
                else:
                    per_tool[call.tool_name]["failed"] += 1

        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0

        breakdowns = [
            ToolBreakdown(
                tool_name=name,
                total_calls=data["total"],
                successful_calls=data["success"],
                failed_calls=data["failed"],
                success_rate=data["success"] / data["total"] if data["total"] > 0 else 0.0,
            )
            for name, data in per_tool.items()
        ]

        return ToolMetrics(
            total_calls=total,
            successful_calls=successful,
            failed_calls=failed,
            success_rate=success_rate,
            per_tool=breakdowns,
        )
