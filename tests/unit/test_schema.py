"""Unit tests for agent_eval.tracer.schema."""

from __future__ import annotations

import uuid
from datetime import UTC

from agent_eval.tracer.schema import (
    HallucinationFlag,
    RunSummary,
    TokenCount,
    ToolCall,
    Trace,
    Turn,
)


class TestTokenCount:
    def test_total_property(self) -> None:
        tc = TokenCount(prompt_tokens=30, completion_tokens=70)
        assert tc.total == 100

    def test_default_values(self) -> None:
        tc = TokenCount()
        assert tc.prompt_tokens == 0
        assert tc.completion_tokens == 0
        assert tc.total == 0

    def test_total_with_zeros(self) -> None:
        tc = TokenCount(prompt_tokens=0, completion_tokens=0)
        assert tc.total == 0


class TestHallucinationFlag:
    def test_creation(self) -> None:
        flag = HallucinationFlag(
            argument_name="url",
            expected="valid URL string",
            received="not-a-url",
            confidence=0.95,
            method="schema",
        )
        assert flag.argument_name == "url"
        assert flag.confidence == 0.95
        assert flag.method == "schema"

    def test_all_methods(self) -> None:
        for method in ("schema", "semantic", "llm_judge"):
            flag = HallucinationFlag(
                argument_name="x",
                expected="something",
                received="other",
                confidence=0.5,
                method=method,
            )
            assert flag.method == method

    def test_received_can_be_any_type(self) -> None:
        flag = HallucinationFlag(
            argument_name="count",
            expected="positive integer",
            received={"nested": "dict"},
            confidence=0.8,
            method="llm_judge",
        )
        assert flag.received == {"nested": "dict"}


class TestToolCall:
    def test_default_call_id_is_uuid(self, sample_tool_call: ToolCall) -> None:
        parsed = uuid.UUID(sample_tool_call.call_id)
        assert str(parsed) == sample_tool_call.call_id

    def test_creation(self, sample_tool_call: ToolCall) -> None:
        assert sample_tool_call.tool_name == "search_web"
        assert sample_tool_call.input_args == {"query": "test query"}
        assert sample_tool_call.output == "some results"
        assert sample_tool_call.success is True
        assert sample_tool_call.latency_ms == 100

    def test_hallucination_flags_default_empty(self, sample_tool_call: ToolCall) -> None:
        assert sample_tool_call.hallucination_flags == []

    def test_with_hallucination_flag(self) -> None:
        flag = HallucinationFlag(
            argument_name="query",
            expected="string",
            received=None,
            confidence=0.7,
            method="schema",
        )
        call = ToolCall(
            tool_name="search",
            input_args={"query": None},
            output=None,
            success=False,
            latency_ms=50,
            hallucination_flags=[flag],
        )
        assert len(call.hallucination_flags) == 1
        assert call.hallucination_flags[0].argument_name == "query"

    def test_failed_tool_call(self) -> None:
        call = ToolCall(
            tool_name="delete_file",
            input_args={"path": "/nonexistent"},
            output=None,
            success=False,
            latency_ms=10,
        )
        assert call.success is False
        assert call.output is None


class TestTurn:
    def test_creation(self, sample_turn: Turn) -> None:
        assert sample_turn.turn_id == 0
        assert sample_turn.role == "assistant"
        assert sample_turn.content == "I'll search for that"
        assert sample_turn.latency_ms == 150

    def test_token_count(self, sample_turn: Turn) -> None:
        assert sample_turn.tokens.prompt_tokens == 50
        assert sample_turn.tokens.completion_tokens == 20
        assert sample_turn.tokens.total == 70

    def test_tool_calls_list(self, sample_turn: Turn) -> None:
        assert len(sample_turn.tool_calls) == 1

    def test_default_latency_zero(self) -> None:
        turn = Turn(turn_id=1, role="user", content="hello")
        assert turn.latency_ms == 0

    def test_default_tool_calls_empty(self) -> None:
        turn = Turn(turn_id=2, role="system", content="You are helpful.")
        assert turn.tool_calls == []

    def test_roles(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            turn = Turn(turn_id=0, role=role, content="test")
            assert turn.role == role


class TestTrace:
    def test_default_run_id_is_uuid(self, sample_trace: Trace) -> None:
        parsed = uuid.UUID(sample_trace.run_id)
        assert str(parsed) == sample_trace.run_id

    def test_default_created_at_is_utc(self, sample_trace: Trace) -> None:
        assert sample_trace.created_at.tzinfo is not None
        assert sample_trace.created_at.tzinfo == UTC

    def test_schema_version_default(self, sample_trace: Trace) -> None:
        assert sample_trace.schema_version == "1"

    def test_model_and_task(self, sample_trace: Trace) -> None:
        assert sample_trace.model == "claude-sonnet-4-6"
        assert sample_trace.task == "test task"

    def test_turns_list(self, sample_trace: Trace) -> None:
        assert len(sample_trace.turns) == 1

    def test_summary_defaults_none(self, sample_trace: Trace) -> None:
        assert sample_trace.summary is None

    def test_agent_config_defaults_empty(self) -> None:
        trace = Trace(model="gpt-4o")
        assert trace.agent_config == {}

    def test_task_optional(self) -> None:
        trace = Trace(model="gpt-4o")
        assert trace.task is None

    def test_each_run_id_unique(self) -> None:
        t1 = Trace(model="m")
        t2 = Trace(model="m")
        assert t1.run_id != t2.run_id

    def test_serialization_roundtrip(self, sample_trace: Trace) -> None:
        json_str = sample_trace.model_dump_json(indent=2)
        restored = Trace.model_validate_json(json_str)
        assert restored.run_id == sample_trace.run_id
        assert restored.model == sample_trace.model
        assert restored.task == sample_trace.task
        assert len(restored.turns) == len(sample_trace.turns)

    def test_serialization_roundtrip_with_hallucination(self) -> None:
        flag = HallucinationFlag(
            argument_name="arg",
            expected="str",
            received=42,
            confidence=0.9,
            method="schema",
        )
        call = ToolCall(
            tool_name="tool",
            input_args={"arg": 42},
            output=None,
            success=False,
            latency_ms=5,
            hallucination_flags=[flag],
        )
        turn = Turn(turn_id=0, role="assistant", content="using tool", tool_calls=[call])
        trace = Trace(model="claude-sonnet-4-6", turns=[turn])
        json_str = trace.model_dump_json(indent=2)
        restored = Trace.model_validate_json(json_str)
        assert len(restored.turns[0].tool_calls[0].hallucination_flags) == 1
        assert restored.turns[0].tool_calls[0].hallucination_flags[0].confidence == 0.9

    def test_with_run_summary(self) -> None:
        summary = RunSummary(
            turn_count=2,
            total_tool_calls=3,
            successful_tool_calls=2,
            failed_tool_calls=1,
            tool_success_rate=0.667,
            hallucination_rate=0.0,
            total_latency_ms=500,
            p50_turn_latency_ms=250,
            p95_turn_latency_ms=480,
            estimated_cost_usd=0.002,
        )
        trace = Trace(model="gpt-4o", summary=summary)
        assert trace.summary is not None
        assert trace.summary.turn_count == 2

    def test_model_dump_json_and_validate(self) -> None:
        trace = Trace(model="test-model", task="unit test", agent_config={"temperature": 0.7})
        json_str = trace.model_dump_json()
        restored = Trace.model_validate_json(json_str)
        assert restored.agent_config == {"temperature": 0.7}
