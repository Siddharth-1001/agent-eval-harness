"""LangGraph / LangChain tracing adapter."""

from __future__ import annotations

import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


class _EvalCallbackHandler:
    """
    A callback handler that captures tool start/end and LLM end events.
    Inherits from BaseCallbackHandler at runtime (lazy import).
    """

    def __init__(self, collector: TraceCollector) -> None:
        self._collector = collector
        self._tool_start_times: dict[str, float] = {}
        self._tool_inputs: dict[str, dict[str, Any]] = {}
        self._current_turn_id: int | None = None

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # Extract text from generations
        content = ""
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    text = getattr(gen, "text", None)
                    if text:
                        content += text
        except Exception:
            content = str(response)
        turn_id = self._collector.start_turn("assistant", content)
        self._collector.end_turn(turn_id)
        self._current_turn_id = turn_id

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id) if run_id is not None else input_str
        self._tool_start_times[run_key] = time.monotonic()
        # Try to parse input as dict, otherwise store as {"input": input_str}
        try:
            import json

            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                self._tool_inputs[run_key] = parsed
            else:
                self._tool_inputs[run_key] = {"input": input_str}
        except Exception:
            self._tool_inputs[run_key] = {"input": input_str}

    def on_tool_end(
        self,
        output: str,
        run_id: Any = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id) if run_id is not None else ""
        start_time = self._tool_start_times.pop(run_key, time.monotonic())
        latency_ms = int((time.monotonic() - start_time) * 1000)
        input_args = self._tool_inputs.pop(run_key, {})

        tool_call = ToolCall(
            tool_name=name or "unknown",
            input_args=input_args,
            output=output,
            success=True,
            latency_ms=latency_ms,
        )

        # Attach to current assistant turn or create a tool turn
        if self._current_turn_id is not None:
            self._collector.record_tool_call(self._current_turn_id, tool_call)
        else:
            turn_id = self._collector.start_turn("tool", output)
            self._collector.record_tool_call(turn_id, tool_call)
            self._collector.end_turn(turn_id, latency_ms=latency_ms)

    def on_tool_error(self, error: Exception, run_id: Any = None, **kwargs: Any) -> None:
        run_key = str(run_id) if run_id is not None else ""
        start_time = self._tool_start_times.pop(run_key, time.monotonic())
        latency_ms = int((time.monotonic() - start_time) * 1000)
        input_args = self._tool_inputs.pop(run_key, {})

        tool_call = ToolCall(
            tool_name="unknown",
            input_args=input_args,
            output=str(error),
            success=False,
            latency_ms=latency_ms,
        )
        if self._current_turn_id is not None:
            self._collector.record_tool_call(self._current_turn_id, tool_call)
        else:
            turn_id = self._collector.start_turn("tool", str(error))
            self._collector.record_tool_call(turn_id, tool_call)
            self._collector.end_turn(turn_id, latency_ms=latency_ms)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        pass


def _make_handler(collector: TraceCollector) -> _EvalCallbackHandler:
    """
    Build a BaseCallbackHandler subclass at runtime (lazy import).
    Falls back to a plain _EvalCallbackHandler if langchain is not installed.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler

        class _RuntimeHandler(_EvalCallbackHandler, BaseCallbackHandler):
            def __init__(self, c: TraceCollector) -> None:
                BaseCallbackHandler.__init__(self)
                _EvalCallbackHandler.__init__(self, c)

        return _RuntimeHandler(collector)
    except ImportError:
        return _EvalCallbackHandler(collector)


class LangGraphTracer:
    """
    Context manager that wires eval-harness tracing into a LangGraph graph.

    Usage::

        async with LangGraphTracer(task="my-task") as tracer:
            result = await graph.ainvoke(state, config=tracer.langgraph_config)
    """

    def __init__(
        self,
        task: str | None = None,
        model: str = "unknown",
        writer_config: TraceWriterConfig | None = None,
    ) -> None:
        self._task = task
        self._model = model
        self._writer_config = writer_config
        self._collector = TraceCollector(model=model, task=task)
        self._handler = _make_handler(self._collector)

    @property
    def langgraph_config(self) -> dict[str, Any]:
        """Returns {"callbacks": [self._handler]} — pass to graph.ainvoke()."""
        return {"callbacks": [self._handler]}

    async def __aenter__(self) -> LangGraphTracer:
        return self

    async def __aexit__(self, *args: Any) -> None:
        trace = await self._collector.async_finalize()
        writer = TraceWriter(self._writer_config)
        await writer.async_write(trace)

    def __enter__(self) -> LangGraphTracer:
        return self

    def __exit__(self, *args: Any) -> None:
        trace = self._collector.finalize()
        writer = TraceWriter(self._writer_config)
        writer.write(trace)
