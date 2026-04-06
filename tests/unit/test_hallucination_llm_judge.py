"""Tests for hallucination detection in llm_judge mode with mocked judge."""

from __future__ import annotations

import pytest

from agent_eval.metrics.hallucination import (
    HallucinationConfig,
    HallucinationDetector,
    ToolHallucinationConfig,
)
from agent_eval.tracer.schema import HallucinationFlag, ToolCall, Trace, Turn


def make_call(tool_name: str, input_args: dict) -> ToolCall:
    return ToolCall(tool_name=tool_name, input_args=input_args, success=True, latency_ms=10)


def make_flag(arg_name: str) -> HallucinationFlag:
    return HallucinationFlag(
        argument_name=arg_name,
        expected="some expected value",
        received="some received value",
        confidence=0.85,
        method="llm_judge",
    )


class MockLLMJudge:
    """Mock LLM judge that returns pre-configured flags."""

    def __init__(self, flags: list[HallucinationFlag]) -> None:
        self._flags = flags
        self.calls: list[dict] = []

    async def judge(
        self, tool_call: ToolCall, context: str, model: str, sensitivity: float
    ) -> list[HallucinationFlag]:
        self.calls.append(
            {
                "tool_call": tool_call,
                "context": context,
                "model": model,
                "sensitivity": sensitivity,
            }
        )
        return self._flags


@pytest.mark.asyncio
async def test_mock_judge_returning_flags_appear_in_result():
    expected_flags = [make_flag("arg1"), make_flag("arg2")]
    mock_judge = MockLLMJudge(expected_flags)

    tool_config = ToolHallucinationConfig(
        mode="llm_judge", judge_model="claude-haiku-4-5", sensitivity=0.7
    )
    call = make_call("analyze", {"arg1": "bad_value", "arg2": "also_bad"})
    turn = Turn(turn_id=0, role="assistant", content="some context", tool_calls=[call])
    config = HallucinationConfig(tools={"analyze": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config, llm_judge=mock_judge)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 2
    assert len(call.hallucination_flags) == 2
    assert "llm_judge" in metrics.flags_by_method
    assert metrics.flags_by_method["llm_judge"] == 2


@pytest.mark.asyncio
async def test_mock_judge_returning_empty_no_flags():
    mock_judge = MockLLMJudge([])

    tool_config = ToolHallucinationConfig(mode="llm_judge")
    call = make_call("analyze", {"arg1": "good_value"})
    turn = Turn(turn_id=0, role="assistant", content="context", tool_calls=[call])
    config = HallucinationConfig(tools={"analyze": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config, llm_judge=mock_judge)
    metrics = await detector.analyze(trace)

    assert metrics.total_flags == 0
    assert call.hallucination_flags == []


@pytest.mark.asyncio
async def test_judge_called_with_correct_arguments():
    mock_judge = MockLLMJudge([])

    tool_config = ToolHallucinationConfig(
        mode="llm_judge",
        judge_model="claude-haiku-4-5",
        sensitivity=0.85,
    )
    call = make_call("search", {"query": "test"})
    turn = Turn(
        turn_id=0, role="assistant", content="user asked to search for test", tool_calls=[call]
    )
    config = HallucinationConfig(tools={"search": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config, llm_judge=mock_judge)
    await detector.analyze(trace)

    assert len(mock_judge.calls) == 1
    call_args = mock_judge.calls[0]
    assert call_args["tool_call"] is call
    assert call_args["context"] == "user asked to search for test"
    assert call_args["model"] == "claude-haiku-4-5"
    assert call_args["sensitivity"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_llm_judge_mode_also_runs_schema_check():
    """llm_judge mode still runs schema check first."""
    mock_judge = MockLLMJudge([])

    schema = {
        "required": ["required_arg"],
        "properties": {"required_arg": {"type": "string"}},
    }
    tool_config = ToolHallucinationConfig(
        mode="llm_judge",
        json_schema=schema,
        judge_model="claude-haiku-4-5",
    )
    call = make_call("tool", {})  # missing required_arg
    turn = Turn(turn_id=0, role="assistant", content="content", tool_calls=[call])
    config = HallucinationConfig(tools={"tool": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config, llm_judge=mock_judge)
    metrics = await detector.analyze(trace)

    # Schema flag from missing required_arg
    assert metrics.total_flags >= 1
    schema_flags = [f for f in call.hallucination_flags if f.method == "schema"]
    assert len(schema_flags) == 1
    assert schema_flags[0].argument_name == "required_arg"


@pytest.mark.asyncio
async def test_hallucination_rate_computed_correctly_with_judge():
    mock_judge = MockLLMJudge([make_flag("bad_arg")])

    tool_config = ToolHallucinationConfig(mode="llm_judge")
    calls = [
        make_call("tool", {"x": 1}),
        make_call("tool", {"x": 2}),
    ]
    turn = Turn(turn_id=0, role="assistant", content="content", tool_calls=calls)
    config = HallucinationConfig(tools={"tool": tool_config})
    trace = Trace(model="claude-sonnet-4-6", turns=[turn])

    detector = HallucinationDetector(config, llm_judge=mock_judge)
    metrics = await detector.analyze(trace)

    # 2 flags (one per call), 2 total calls => rate = 1.0
    assert metrics.total_flags == 2
    assert metrics.hallucination_rate == pytest.approx(1.0)


# ── _build_judge_prompt ───────────────────────────────────────────────────────


def test_build_judge_prompt_contains_tool_name():
    from agent_eval.metrics.hallucination import _build_judge_prompt

    call = make_call("search_db", {"query": "hello", "limit": 10})
    prompt = _build_judge_prompt(call, "Find recent records")
    assert "search_db" in prompt
    assert "query" in prompt
    assert "Find recent records" in prompt


def test_build_judge_prompt_no_context():
    from agent_eval.metrics.hallucination import _build_judge_prompt

    call = make_call("tool", {})
    prompt = _build_judge_prompt(call, "")
    assert "No context provided" in prompt


def test_build_judge_prompt_no_args():
    from agent_eval.metrics.hallucination import _build_judge_prompt

    call = make_call("tool", {})
    prompt = _build_judge_prompt(call, "ctx")
    assert "(no arguments)" in prompt


# ── _parse_judge_response ─────────────────────────────────────────────────────


def test_parse_judge_response_none_returns_empty():
    from agent_eval.metrics.hallucination import _parse_judge_response

    call = make_call("tool", {})
    flags = _parse_judge_response("NONE", call, sensitivity=0.7)
    assert flags == []


def test_parse_judge_response_valid_hallucination_line():
    from agent_eval.metrics.hallucination import _parse_judge_response

    raw = "HALLUCINATION: city | expected: a real city | received: Narnia | confidence: 0.9"
    call = make_call("tool", {"city": "Narnia"})
    flags = _parse_judge_response(raw, call, sensitivity=0.7)
    assert len(flags) == 1
    assert flags[0].argument_name == "city"
    assert flags[0].confidence == pytest.approx(0.9)
    assert flags[0].method == "llm_judge"


def test_parse_judge_response_below_sensitivity_filtered():
    from agent_eval.metrics.hallucination import _parse_judge_response

    raw = "HALLUCINATION: city | expected: real | received: Narnia | confidence: 0.5"
    call = make_call("tool", {"city": "Narnia"})
    flags = _parse_judge_response(raw, call, sensitivity=0.8)
    assert flags == []


def test_parse_judge_response_malformed_line_skipped():
    from agent_eval.metrics.hallucination import _parse_judge_response

    raw = "HALLUCINATION: only_one_part"
    call = make_call("tool", {})
    flags = _parse_judge_response(raw, call, sensitivity=0.5)
    assert flags == []


def test_parse_judge_response_multiple_lines():
    from agent_eval.metrics.hallucination import _parse_judge_response

    raw = (
        "HALLUCINATION: a | expected: x | received: y | confidence: 0.9\n"
        "HALLUCINATION: b | expected: p | received: q | confidence: 0.8\n"
    )
    call = make_call("tool", {"a": "y", "b": "q"})
    flags = _parse_judge_response(raw, call, sensitivity=0.7)
    assert len(flags) == 2
    assert {f.argument_name for f in flags} == {"a", "b"}


# ── AnthropicLLMJudge (mocked) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_llm_judge_returns_flags():
    """AnthropicLLMJudge calls the Anthropic API and parses the structured response."""
    import sys
    from unittest.mock import AsyncMock, MagicMock

    mock_content_block = MagicMock()
    mock_content_block.text = (
        "HALLUCINATION: city | expected: real city | received: Narnia | confidence: 0.95"
    )
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.AsyncAnthropic.return_value = mock_client

    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "anthropic", mock_anthropic_module)
        from agent_eval.metrics.hallucination import AnthropicLLMJudge

        judge = AnthropicLLMJudge(api_key="test-key", model="claude-haiku-4-5")
        call = make_call("lookup", {"city": "Narnia"})
        flags = await judge.judge(call, "Look up city details", "claude-haiku-4-5", 0.7)

    assert len(flags) == 1
    assert flags[0].argument_name == "city"
    assert flags[0].method == "llm_judge"


@pytest.mark.asyncio
async def test_anthropic_llm_judge_import_error():
    """AnthropicLLMJudge raises ImportError when anthropic is not installed."""
    from unittest.mock import patch

    from agent_eval.metrics.hallucination import AnthropicLLMJudge

    with patch.dict("sys.modules", {"anthropic": None}):
        judge = AnthropicLLMJudge()
        call = make_call("tool", {})
        with pytest.raises(ImportError, match="anthropic"):
            await judge.judge(call, "", "claude-haiku-4-5", 0.7)


# ── OpenAILLMJudge (mocked) ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_llm_judge_returns_flags():
    """OpenAILLMJudge calls the OpenAI API and parses the structured response."""
    import sys
    from unittest.mock import AsyncMock, MagicMock

    mock_message = MagicMock()
    mock_message.content = (
        "HALLUCINATION: country | expected: real country | received: Mordor | confidence: 0.88"
    )
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_openai_module = MagicMock()
    mock_openai_module.AsyncOpenAI.return_value = mock_client

    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "openai", mock_openai_module)
        from agent_eval.metrics.hallucination import OpenAILLMJudge

        judge = OpenAILLMJudge(api_key="test-key", model="gpt-4o-mini")
        call = make_call("geocode", {"country": "Mordor"})
        flags = await judge.judge(call, "Geocode the location", "gpt-4o-mini", 0.7)

    assert len(flags) == 1
    assert flags[0].argument_name == "country"
    assert flags[0].method == "llm_judge"


@pytest.mark.asyncio
async def test_openai_llm_judge_import_error():
    """OpenAILLMJudge raises ImportError when openai is not installed."""
    from unittest.mock import patch

    from agent_eval.metrics.hallucination import OpenAILLMJudge

    with patch.dict("sys.modules", {"openai": None}):
        judge = OpenAILLMJudge()
        call = make_call("tool", {})
        with pytest.raises(ImportError, match="openai"):
            await judge.judge(call, "", "gpt-4o-mini", 0.7)
