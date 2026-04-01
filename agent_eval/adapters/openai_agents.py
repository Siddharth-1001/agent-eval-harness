"""OpenAI Agents SDK tracing decorator."""

from __future__ import annotations

import contextvars
import functools
import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig

# ── Context variable ──────────────────────────────────────────────────────────
# Carries the active TraceCollector into nested Runner.run() calls so that
# EvalRunHooks can be injected automatically without changing user code.
_ACTIVE_COLLECTOR: contextvars.ContextVar[TraceCollector | None] = contextvars.ContextVar(
    "_agent_eval_collector", default=None
)

_RUNNER_PATCHED = False


def _patch_runner_once() -> None:
    """
    Monkey-patch Runner.run() once (lazily) to inject EvalRunHooks from the
    ContextVar when a collector is active.  Safe to call repeatedly.
    """
    global _RUNNER_PATCHED
    if _RUNNER_PATCHED:
        return
    _RUNNER_PATCHED = True
    try:
        from openai_agents import Runner  # type: ignore[import-untyped]
    except Exception:
        return

    original_run = getattr(Runner, "run", None)
    if original_run is None:
        return
    # Unwrap the classmethod to get the underlying function
    original_func = getattr(original_run, "__func__", None)
    if original_func is None:
        return

    @functools.wraps(original_func)
    async def _patched_func(cls: Any, starting_agent: Any, input: Any, **kwargs: Any) -> Any:
        collector = _ACTIVE_COLLECTOR.get()
        if collector is not None and "hooks" not in kwargs:
            kwargs["hooks"] = EvalRunHooks(collector)
        return await original_func(cls, starting_agent, input, **kwargs)

    Runner.run = classmethod(_patched_func)


class EvalRunHooks:
    """
    Implements the openai-agents ``RunHooks`` protocol.

    Records tool calls into a :class:`TraceCollector`.  Can be used directly::

        hooks = EvalRunHooks(collector)
        result = await Runner.run(agent, input_text, hooks=hooks)

    Or automatically via :func:`trace_openai_agent`, which injects it into
    every ``Runner.run()`` call made inside the decorated function.
    """

    def __init__(self, collector: TraceCollector) -> None:
        self._collector = collector
        # FIFO list of (tool_name, start_time) for in-flight sequential tool calls
        self._active: list[tuple[str, float]] = []

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        pass

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        pass

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        pass

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        self._active.append((tool_name, time.monotonic()))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
        if not self._active:
            return
        tool_name, start_time = self._active.pop(0)  # FIFO
        latency_ms = int((time.monotonic() - start_time) * 1000)
        tool_call = ToolCall(
            tool_name=tool_name,
            input_args={},
            output=str(result),
            success=True,
            latency_ms=latency_ms,
        )
        turn_id = await self._collector.async_start_turn("tool", str(result)[:1000])
        await self._collector.async_record_tool_call(turn_id, tool_call)
        await self._collector.async_end_turn(turn_id, latency_ms=latency_ms)


def trace_openai_agent(
    task: str | None = None,
    model: str = "unknown",
    writer_config: TraceWriterConfig | None = None,
) -> Any:
    """
    Decorator that wraps an async function calling Runner.run() and saves a trace.

    Automatically injects :class:`EvalRunHooks` into every ``Runner.run()`` call
    made inside the decorated function, so tool calls are captured without any
    changes to user code.

    Usage::

        @trace_openai_agent(task="my-task", model="gpt-4o")
        async def run_agent(input_text: str) -> str:
            result = await Runner.run(agent, input_text)
            return result.final_output
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import contextlib

            with contextlib.suppress(ImportError):
                import openai_agents  # noqa: F401

            # Patch Runner.run once so it picks up hooks from the ContextVar
            _patch_runner_once()

            collector = TraceCollector(model=model, task=task)

            # Propagate the collector into nested async calls via ContextVar
            token = _ACTIVE_COLLECTOR.set(collector)

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
                _ACTIVE_COLLECTOR.reset(token)
                trace = await collector.async_finalize()
                writer = TraceWriter(writer_config)
                await writer.async_write(trace)

        return wrapper

    return decorator
