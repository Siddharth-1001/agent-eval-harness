"""
OpenAI Agents SDK example — task planning agent.
Runs fully offline using MockLLM. No API key required.

# REAL USAGE:
#   from openai_agents import Agent, Runner
#   agent = Agent(name="Planner", model="gpt-4o", instructions="Plan tasks step by step.")
#   @trace_openai_agent(task="task-planning", model="gpt-4o")
#   async def run_agent(input_text: str) -> str:
#       result = await Runner.run(agent, input_text)
#       return result.final_output
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig
from examples.mock_llm import MockLLM


def list_tasks(project: str) -> str:
    """Mock task-listing tool."""
    return f"[Mock tasks for: {project}] — Task A, Task B, Task C."


async def run_planning_agent(goal: str) -> str:
    llm = MockLLM()
    collector = TraceCollector(model=llm.model, task="task-planning")
    writer = TraceWriter(TraceWriterConfig(output_dir=Path("./traces")))

    # Turn 1: user goal
    turn_id = collector.start_turn("user", goal)
    collector.end_turn(turn_id, latency_ms=0)

    # Turn 2: assistant lists tasks
    response = llm.invoke(goal)
    turn_id = collector.start_turn("assistant", response.content)

    tool_call = ToolCall(
        tool_name="list_tasks",
        input_args={"project": goal},
        output=list_tasks(goal),
        success=True,
        latency_ms=35,
    )
    collector.record_tool_call(turn_id, tool_call)
    collector.end_turn(turn_id, latency_ms=120)

    # Turn 3: assistant produces a plan
    plan = llm.invoke("plan")
    turn_id = collector.start_turn("assistant", plan.content)
    collector.end_turn(turn_id, latency_ms=90)

    trace = collector.finalize()
    path = writer.write(trace)
    print(f"Trace saved → {path}")
    print(f"Run ID: {trace.run_id}")
    return plan.content


if __name__ == "__main__":
    result = asyncio.run(run_planning_agent("Build a REST API"))
    print(f"\nResult: {result}")
