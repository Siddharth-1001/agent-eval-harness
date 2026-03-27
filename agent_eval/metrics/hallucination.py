"""Hallucination detection in three modes: schema, semantic, llm_judge."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from pydantic import BaseModel

from agent_eval.tracer.schema import HallucinationFlag, ToolCall, Trace


class ToolHallucinationConfig(BaseModel):
    """Per-tool hallucination detection configuration."""

    mode: str = "schema"  # "schema" | "semantic" | "llm_judge"
    schema: dict[str, Any] | None = None  # JSON schema for the tool's input_args
    value_sets: dict[str, list[Any]] | None = None  # for semantic mode
    judge_model: str = "claude-haiku-4-5"  # for llm_judge mode
    sensitivity: float = 0.7  # for llm_judge mode, confidence threshold


class HallucinationConfig(BaseModel):
    """Global hallucination detection configuration."""

    tools: dict[str, ToolHallucinationConfig] = {}
    default_mode: str = "schema"


class LLMJudge(Protocol):
    """Protocol for LLM judge implementations (mockable in tests)."""

    async def judge(
        self, tool_call: ToolCall, context: str, model: str, sensitivity: float
    ) -> list[HallucinationFlag]: ...


class _DefaultLLMJudge:
    """No-op LLM judge (for testing without real API access)."""

    async def judge(
        self, tool_call: ToolCall, context: str, model: str, sensitivity: float
    ) -> list[HallucinationFlag]:
        return []


def _schema_check(tool_call: ToolCall, schema: dict[str, Any] | None) -> list[HallucinationFlag]:
    """Mode 1: Schema validation. Check input_args against JSON schema."""
    flags: list[HallucinationFlag] = []
    if not schema:
        return flags

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # Check required fields
    for field in required:
        if field not in tool_call.input_args:
            flags.append(
                HallucinationFlag(
                    argument_name=field,
                    expected=f"required field '{field}'",
                    received=None,
                    confidence=1.0,
                    method="schema",
                )
            )

    # Check types
    for arg_name, value in tool_call.input_args.items():
        if arg_name in properties:
            prop_schema = properties[arg_name]
            expected_type = prop_schema.get("type")
            if expected_type and not _type_matches(value, expected_type):
                flags.append(
                    HallucinationFlag(
                        argument_name=arg_name,
                        expected=f"type {expected_type}",
                        received=value,
                        confidence=1.0,
                        method="schema",
                    )
                )

    return flags


def _type_matches(value: Any, expected_type: str) -> bool:
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    py_type = type_map.get(expected_type)
    if py_type is None:
        return True  # Unknown type — don't flag
    return isinstance(value, py_type)


def _semantic_check(
    tool_call: ToolCall, value_sets: dict[str, list[Any]]
) -> list[HallucinationFlag]:
    """Mode 2: Semantic validation. Check argument values against allowed sets."""
    flags: list[HallucinationFlag] = []
    for arg_name, allowed_values in value_sets.items():
        if arg_name not in tool_call.input_args:
            continue
        received = tool_call.input_args[arg_name]
        if received not in allowed_values:
            flags.append(
                HallucinationFlag(
                    argument_name=arg_name,
                    expected=f"one of {allowed_values}",
                    received=received,
                    confidence=0.9,
                    method="semantic",
                )
            )
    return flags


class HallucinationMetrics(BaseModel):
    total_flags: int
    hallucination_rate: float  # flags / total tool calls
    flags_by_tool: dict[str, int]
    flags_by_method: dict[str, int]


class HallucinationDetector:
    """Detects hallucinations in tool calls using three configurable modes."""

    def __init__(
        self, config: HallucinationConfig | None = None, llm_judge: LLMJudge | None = None
    ) -> None:
        self.config = config or HallucinationConfig()
        self._llm_judge: LLMJudge = llm_judge or _DefaultLLMJudge()

    def _get_tool_config(self, tool_name: str) -> ToolHallucinationConfig:
        if tool_name in self.config.tools:
            return self.config.tools[tool_name]
        return ToolHallucinationConfig(mode=self.config.default_mode)

    async def _check_tool_call(self, call: ToolCall, turn_content: str) -> list[HallucinationFlag]:
        tool_config = self._get_tool_config(call.tool_name)
        flags: list[HallucinationFlag] = []

        if tool_config.mode == "schema":
            flags.extend(_schema_check(call, tool_config.schema))
        elif tool_config.mode == "semantic":
            flags.extend(_schema_check(call, tool_config.schema))
            if tool_config.value_sets:
                flags.extend(_semantic_check(call, tool_config.value_sets))
        elif tool_config.mode == "llm_judge":
            flags.extend(_schema_check(call, tool_config.schema))
            judge_flags = await self._llm_judge.judge(
                call, turn_content, tool_config.judge_model, tool_config.sensitivity
            )
            flags.extend(judge_flags)

        return flags

    async def analyze(self, trace: Trace) -> HallucinationMetrics:
        """Run detection on all tool calls in the trace, mutating hallucination_flags in place."""
        total_calls = 0
        total_flags = 0
        flags_by_tool: dict[str, int] = {}
        flags_by_method: dict[str, int] = {}

        tasks = []
        tool_calls_ref: list[ToolCall] = []

        for turn in trace.turns:
            for call in turn.tool_calls:
                total_calls += 1
                tool_calls_ref.append(call)
                tasks.append(self._check_tool_call(call, turn.content))

        if tasks:
            results = await asyncio.gather(*tasks)
            for call, flags in zip(tool_calls_ref, results, strict=True):
                call.hallucination_flags = flags
                count = len(flags)
                total_flags += count
                flags_by_tool[call.tool_name] = flags_by_tool.get(call.tool_name, 0) + count
                for flag in flags:
                    flags_by_method[flag.method] = flags_by_method.get(flag.method, 0) + 1

        hallucination_rate = total_flags / total_calls if total_calls > 0 else 0.0
        return HallucinationMetrics(
            total_flags=total_flags,
            hallucination_rate=hallucination_rate,
            flags_by_tool=flags_by_tool,
            flags_by_method=flags_by_method,
        )
