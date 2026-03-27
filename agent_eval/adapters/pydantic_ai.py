"""PydanticAI tracing adapter."""

from __future__ import annotations

import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


class _WrappedPydanticAgent:
    """Wraps a PydanticAI agent to add eval-harness tracing."""

    def __init__(
        self,
        agent: Any,
        task: str | None = None,
        model: str = "unknown",
        writer_config: TraceWriterConfig | None = None,
    ) -> None:
        self._agent = agent
        self._task = task
        self._model = model
        self._writer_config = writer_config

    def _extract_model_name(self) -> str:
        """Try to pull the model name from the wrapped agent."""
        try:
            m = getattr(self._agent, "model", None)
            if m is not None:
                return str(getattr(m, "model_name", None) or getattr(m, "name", None) or m)
        except Exception:
            pass
        return self._model

    def _extract_tool_calls_from_result(self, result: Any, latency_ms: int) -> list[ToolCall]:
        """Extract tool calls recorded in a RunResult."""
        tool_calls: list[ToolCall] = []
        try:
            all_messages = getattr(result, "all_messages", None)
            messages = all_messages() if callable(all_messages) else all_messages or []

            for msg in messages:
                parts = getattr(msg, "parts", [])
                for part in parts:
                    part_type = type(part).__name__
                    if "ToolCall" in part_type or "tool_call" in part_type.lower():
                        tc = ToolCall(
                            tool_name=str(getattr(part, "tool_name", "unknown")),
                            input_args=dict(getattr(part, "args", {}) or {}),
                            output=None,
                            success=True,
                            latency_ms=latency_ms,
                        )
                        tool_calls.append(tc)
                    elif "ToolReturn" in part_type or "tool_return" in part_type.lower():
                        # Update the output of the last matching tool call
                        tool_name = str(getattr(part, "tool_name", ""))
                        output_val = getattr(part, "content", None)
                        for tc in reversed(tool_calls):
                            if tc.tool_name == tool_name and tc.output is None:
                                tc.output = output_val
                                break
        except Exception:
            pass
        return tool_calls

    def run_sync(self, user_prompt: str, **kwargs: Any) -> Any:
        """Synchronous run with tracing."""
        model_name = self._extract_model_name()
        collector = TraceCollector(model=model_name, task=self._task)

        user_turn_id = collector.start_turn("user", user_prompt)
        collector.end_turn(user_turn_id)

        start = time.monotonic()
        try:
            result = self._agent.run_sync(user_prompt, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)

            output_str = str(getattr(result, "data", result))
            assistant_turn_id = collector.start_turn("assistant", output_str)

            tool_calls = self._extract_tool_calls_from_result(result, latency_ms)
            for tc in tool_calls:
                collector.record_tool_call(assistant_turn_id, tc)

            collector.end_turn(assistant_turn_id, latency_ms=latency_ms)
            return result
        except Exception:
            raise
        finally:
            trace = collector.finalize()
            writer = TraceWriter(self._writer_config)
            writer.write(trace)

    async def run(self, user_prompt: str, **kwargs: Any) -> Any:
        """Async run with tracing."""
        model_name = self._extract_model_name()
        collector = TraceCollector(model=model_name, task=self._task)

        user_turn_id = await collector.async_start_turn("user", user_prompt)
        await collector.async_end_turn(user_turn_id)

        start = time.monotonic()
        try:
            result = await self._agent.run(user_prompt, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)

            output_str = str(getattr(result, "data", result))
            assistant_turn_id = await collector.async_start_turn("assistant", output_str)

            tool_calls = self._extract_tool_calls_from_result(result, latency_ms)
            for tc in tool_calls:
                await collector.async_record_tool_call(assistant_turn_id, tc)

            await collector.async_end_turn(assistant_turn_id, latency_ms=latency_ms)
            return result
        except Exception:
            raise
        finally:
            trace = await collector.async_finalize()
            writer = TraceWriter(self._writer_config)
            await writer.async_write(trace)


def with_eval_harness(
    agent: Any,
    task: str | None = None,
    model: str = "unknown",
    writer_config: TraceWriterConfig | None = None,
) -> _WrappedPydanticAgent:
    """
    Wrap a PydanticAI agent with eval-harness tracing.

    Usage::

        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4o", system_prompt="You are helpful.")
        traced = with_eval_harness(agent, task="my-task")
        result = await traced.run("Hello!")
    """
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        raise ImportError(
            "Install pydantic-ai: pip install 'agent-eval-harness[pydantic-ai]'"
        ) from None

    return _WrappedPydanticAgent(agent, task=task, model=model, writer_config=writer_config)
