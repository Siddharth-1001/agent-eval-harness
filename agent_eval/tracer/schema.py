from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class HallucinationFlag(BaseModel):
    argument_name: str
    expected: str
    received: Any
    confidence: float  # 0.0–1.0
    method: str  # "schema" | "semantic" | "llm_judge"


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    input_args: dict[str, Any]
    output: Any = None
    success: bool
    latency_ms: int
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)


class TokenCount(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Turn(BaseModel):
    turn_id: int
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: int = 0
    tokens: TokenCount = Field(default_factory=TokenCount)


class RunSummary(BaseModel):
    turn_count: int
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_success_rate: float
    hallucination_rate: float
    total_latency_ms: int
    p50_turn_latency_ms: int
    p95_turn_latency_ms: int
    estimated_cost_usd: float


class Trace(BaseModel):
    schema_version: str = "1"
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    task: str | None = None
    agent_config: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    summary: RunSummary | None = None
