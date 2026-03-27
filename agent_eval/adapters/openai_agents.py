"""OpenAI Agents SDK tracing decorator."""

from __future__ import annotations

import functools
import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


def trace_openai_agent(
    task: str | None = None,
    model: str = "unknown",
    writer_config: TraceWriterConfig | None = None,
) -> Any:
    """
    Decorator that wraps an async function calling Runner.run() and saves a trace.

    Usage::

        @trace_openai_agent(task="my-task", model="gpt-4o")
        async def run_agent(input_text: str) -> str:
            result = await Runner.run(agent, input_text)
            return result.final_output
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Lazy import check — not required to use the decorator
            import contextlib

            with contextlib.suppress(ImportError):
                import openai_agents  # noqa: F401

            collector = TraceCollector(model=model, task=task)

            # Record user input turn
            user_input = str(args[0]) if args else str(kwargs)
            user_turn_id = await collector.async_start_turn("user", user_input)
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                latency_ms = int((time.monotonic() - start) * 1000)
                await collector.async_end_turn(user_turn_id, latency_ms=latency_ms)

                # Record assistant output turn
                output_str = str(result)
                assistant_turn_id = await collector.async_start_turn("assistant", output_str)
                await collector.async_end_turn(assistant_turn_id)
                return result
            except Exception:
                latency_ms = int((time.monotonic() - start) * 1000)
                await collector.async_end_turn(user_turn_id, latency_ms=latency_ms)
                raise
            finally:
                trace = await collector.async_finalize()
                writer = TraceWriter(writer_config)
                await writer.async_write(trace)

        return wrapper

    return decorator
