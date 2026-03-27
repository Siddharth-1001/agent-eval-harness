from __future__ import annotations

import asyncio
import time
from typing import Any

from agent_eval.tracer.schema import TokenCount, ToolCall, Trace, Turn


class TraceCollector:
    """Stateful event buffer that accumulates trace data during an agent run."""

    def __init__(
        self,
        model: str,
        task: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._task = task
        self._agent_config = agent_config or {}
        self._turns: list[Turn] = []
        self._lock = asyncio.Lock()
        self._turn_start_times: dict[int, float] = {}

    def start_turn(self, role: str, content: str) -> int:
        """Begin a new turn. Returns the turn_id."""
        turn_id = len(self._turns)
        turn = Turn(turn_id=turn_id, role=role, content=content)
        self._turns.append(turn)
        self._turn_start_times[turn_id] = time.monotonic()
        return turn_id

    def record_tool_call(self, turn_id: int, call: ToolCall) -> None:
        """Record a tool call within a turn."""
        self._turns[turn_id].tool_calls.append(call)

    def end_turn(
        self,
        turn_id: int,
        latency_ms: int | None = None,
        tokens: TokenCount | None = None,
    ) -> None:
        """Finalize a turn with timing and token info."""
        turn = self._turns[turn_id]
        if latency_ms is not None:
            turn.latency_ms = latency_ms
        elif turn_id in self._turn_start_times:
            elapsed = time.monotonic() - self._turn_start_times[turn_id]
            turn.latency_ms = int(elapsed * 1000)
        if tokens is not None:
            turn.tokens = tokens

    def finalize(self) -> Trace:
        """Produce the final Trace object."""
        return Trace(
            model=self._model,
            task=self._task,
            agent_config=self._agent_config,
            turns=self._turns,
        )

    # Async versions for async agent runtimes

    async def async_start_turn(self, role: str, content: str) -> int:
        async with self._lock:
            return self.start_turn(role, content)

    async def async_record_tool_call(self, turn_id: int, call: ToolCall) -> None:
        async with self._lock:
            self.record_tool_call(turn_id, call)

    async def async_end_turn(
        self,
        turn_id: int,
        latency_ms: int | None = None,
        tokens: TokenCount | None = None,
    ) -> None:
        async with self._lock:
            self.end_turn(turn_id, latency_ms, tokens)

    async def async_finalize(self) -> Trace:
        async with self._lock:
            return self.finalize()
