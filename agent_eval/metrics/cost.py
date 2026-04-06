"""Cost calculator using pricing.toml."""

from __future__ import annotations

import functools
import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_eval.tracer.schema import Trace

logger = logging.getLogger("agent_eval.cost")

_PRICING_PATH = Path(__file__).parent.parent / "data" / "pricing.toml"


@functools.cache
def _load_pricing() -> dict[str, Any]:
    with open(_PRICING_PATH, "rb") as f:
        return tomllib.load(f)


class CostPerTurn(BaseModel):
    turn_id: int
    cost_usd: float


class CostMetrics(BaseModel):
    total_usd: float
    cost_per_turn: list[CostPerTurn]
    model: str
    input_tokens: int
    output_tokens: int


class MetricsConfig(BaseModel):
    """Configuration for the metrics computation pipeline."""

    pricing_overrides: dict[str, dict[str, float]] | None = None


class CostCalculator:
    """Calculates token-based cost estimates from a trace."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self.config = config or MetricsConfig()
        self._pricing = _load_pricing()

    def _get_rates(self, model: str) -> tuple[float, float]:
        """Returns (input_per_1m, output_per_1m) for a given model."""
        # Check overrides first
        if self.config.pricing_overrides and model in self.config.pricing_overrides:
            override = self.config.pricing_overrides[model]
            return override.get("input_per_1m", 0.0), override.get("output_per_1m", 0.0)
        models = self._pricing.get("models", {})
        if model in models:
            m = models[model]
            return m.get("input_per_1m", 0.0), m.get("output_per_1m", 0.0)
        # Default: unknown model, zero cost
        return 0.0, 0.0

    async def analyze(self, trace: Trace) -> CostMetrics:
        input_rate, output_rate = self._get_rates(trace.model)
        total_input = sum(turn.tokens.prompt_tokens for turn in trace.turns)
        total_output = sum(turn.tokens.completion_tokens for turn in trace.turns)
        total_usd = (total_input * input_rate + total_output * output_rate) / 1_000_000

        cost_per_turn = []
        for turn in trace.turns:
            turn_cost = (
                turn.tokens.prompt_tokens * input_rate + turn.tokens.completion_tokens * output_rate
            ) / 1_000_000
            cost_per_turn.append(CostPerTurn(turn_id=turn.turn_id, cost_usd=turn_cost))

        return CostMetrics(
            total_usd=total_usd,
            cost_per_turn=cost_per_turn,
            model=trace.model,
            input_tokens=total_input,
            output_tokens=total_output,
        )
