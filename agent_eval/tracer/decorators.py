from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import Trace
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


def trace_agent(
    task: str | None = None,
    model: str = "unknown",
    writer_config: TraceWriterConfig | None = None,
) -> Callable[..., Any]:
    """Decorator that wraps a sync or async agent function and saves a trace on completion."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                collector = TraceCollector(model=model, task=task)
                turn_id = await collector.async_start_turn("user", str(args[0]) if args else "")
                start = time.monotonic()
                try:
                    result = await func(*args, **kwargs)
                    latency_ms = int((time.monotonic() - start) * 1000)
                    await collector.async_end_turn(turn_id, latency_ms=latency_ms)
                    assistant_turn_id = await collector.async_start_turn("assistant", str(result))
                    await collector.async_end_turn(assistant_turn_id)
                    return result
                except Exception:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    await collector.async_end_turn(turn_id, latency_ms=latency_ms)
                    raise
                finally:
                    trace = await collector.async_finalize()
                    writer = TraceWriter(writer_config)
                    await writer.async_write(trace)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                collector = TraceCollector(model=model, task=task)
                turn_id = collector.start_turn("user", str(args[0]) if args else "")
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    latency_ms = int((time.monotonic() - start) * 1000)
                    collector.end_turn(turn_id, latency_ms=latency_ms)
                    assistant_turn_id = collector.start_turn("assistant", str(result))
                    collector.end_turn(assistant_turn_id)
                    return result
                except Exception:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    collector.end_turn(turn_id, latency_ms=latency_ms)
                    raise
                finally:
                    trace = collector.finalize()
                    writer = TraceWriter(writer_config)
                    writer.write(trace)

            return sync_wrapper

    return decorator


class AgentTracer:
    """Context manager for tracing block-based / framework agents."""

    def __init__(
        self,
        task: str | None = None,
        model: str = "unknown",
        writer_config: TraceWriterConfig | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.writer_config = writer_config
        self.collector = TraceCollector(model=model, task=task)
        self._trace: Trace | None = None

    async def __aenter__(self) -> AgentTracer:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        trace = await self.collector.async_finalize()
        self._trace = trace
        writer = TraceWriter(self.writer_config)
        await writer.async_write(trace)

    def __enter__(self) -> AgentTracer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        trace = self.collector.finalize()
        self._trace = trace
        writer = TraceWriter(self.writer_config)
        writer.write(trace)

    @property
    def trace(self) -> Trace | None:
        return self._trace
