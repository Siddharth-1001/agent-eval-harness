"""CrewAI tracing adapter."""

from __future__ import annotations

import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


class EvalHarnessCrew:
    """
    A wrapper around a CrewAI Crew that adds eval-harness tracing.

    Usage::

        crew = EvalHarnessCrew(
            agents=[agent1, agent2],
            tasks=[task1, task2],
            task="my-eval-task",
        )
        result = crew.kickoff()
    """

    def __init__(
        self,
        agents: list[Any],
        tasks: list[Any],
        task: str | None = None,
        model: str = "unknown",
        writer_config: TraceWriterConfig | None = None,
        **crew_kwargs: Any,
    ) -> None:
        try:
            from crewai import Crew
        except ImportError:
            raise ImportError("Install crewai: pip install 'agent-eval-harness[crewai]'") from None

        self._task = task
        self._model = model
        self._writer_config = writer_config
        self._collector = TraceCollector(model=model, task=task)

        # Build a step_callback that records tool calls
        original_step_callback = crew_kwargs.pop("step_callback", None)

        def _step_callback(step_output: Any) -> None:
            self._record_step(step_output)
            if original_step_callback is not None:
                original_step_callback(step_output)

        self._crew = Crew(
            agents=agents,
            tasks=tasks,
            step_callback=_step_callback,
            **crew_kwargs,
        )

    def _record_step(self, step_output: Any) -> None:
        """Record a single CrewAI step as a turn with optional tool call."""
        try:
            # AgentAction has .tool, .tool_input, .text attributes
            tool_name = getattr(step_output, "tool", None)
            tool_input = getattr(step_output, "tool_input", None)
            text = getattr(step_output, "text", str(step_output))

            turn_id = self._collector.start_turn("assistant", str(text))

            if tool_name:
                input_args: dict[str, Any] = {}
                if isinstance(tool_input, dict):
                    input_args = tool_input
                elif tool_input is not None:
                    input_args = {"input": str(tool_input)}

                tc = ToolCall(
                    tool_name=str(tool_name),
                    input_args=input_args,
                    output=None,
                    success=True,
                    latency_ms=0,
                )
                self._collector.record_tool_call(turn_id, tc)

            self._collector.end_turn(turn_id)
        except Exception:
            pass  # Never let tracing break the agent

    def kickoff(self) -> Any:
        """Run the crew synchronously and save a trace."""
        start = time.monotonic()
        user_turn_id = self._collector.start_turn("user", "kickoff")
        try:
            result = self._crew.kickoff()
            latency_ms = int((time.monotonic() - start) * 1000)
            self._collector.end_turn(user_turn_id, latency_ms=latency_ms)

            assistant_turn_id = self._collector.start_turn("assistant", str(result))
            self._collector.end_turn(assistant_turn_id)
            return result
        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._collector.end_turn(user_turn_id, latency_ms=latency_ms)
            raise
        finally:
            trace = self._collector.finalize()
            writer = TraceWriter(self._writer_config)
            writer.write(trace)

    async def akickoff(self) -> Any:
        """Run the crew asynchronously and save a trace."""
        start = time.monotonic()
        user_turn_id = await self._collector.async_start_turn("user", "kickoff")
        try:
            result = await self._crew.akickoff()
            latency_ms = int((time.monotonic() - start) * 1000)
            await self._collector.async_end_turn(user_turn_id, latency_ms=latency_ms)

            assistant_turn_id = await self._collector.async_start_turn("assistant", str(result))
            await self._collector.async_end_turn(assistant_turn_id)
            return result
        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            await self._collector.async_end_turn(user_turn_id, latency_ms=latency_ms)
            raise
        finally:
            trace = await self._collector.async_finalize()
            writer = TraceWriter(self._writer_config)
            await writer.async_write(trace)
