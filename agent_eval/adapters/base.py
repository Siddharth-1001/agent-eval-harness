"""Structural protocol for all framework adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_eval.tracer.schema import ToolCall, Turn


@runtime_checkable
class AgentAdapter(Protocol):
    """
    Structural protocol — any class implementing these methods is a valid adapter.
    No inheritance required. True zero-friction integration via structural typing.
    """

    def extract_model(self, run_output: object) -> str: ...

    def extract_turns(self, run_output: object) -> list[Turn]: ...

    def extract_tool_calls(self, turn: object) -> list[ToolCall]: ...
