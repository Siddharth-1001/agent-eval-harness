"""
CrewAI example — content creation crew.
Runs fully offline using MockLLM. No API key required.

# REAL USAGE:
#   from crewai import Agent, Task, Crew
#   from agent_eval.adapters.crewai import EvalHarnessCrew
#   researcher = Agent(role="Researcher", goal="Find information", llm=real_llm)
#   writer = Agent(role="Writer", goal="Write content", llm=real_llm)
#   task1 = Task(description="Research the topic", agent=researcher)
#   task2 = Task(description="Write a blog post", agent=writer)
#   crew = EvalHarnessCrew(agents=[researcher, writer], tasks=[task1, task2], task="content-creation")
#   result = crew.kickoff()
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig
from examples.mock_llm import MockLLM


def research_topic(topic: str) -> str:
    """Mock research tool."""
    return f"[Mock research on: {topic}] — Key finding: This is an important topic."


def write_content(research: str) -> str:
    """Mock content writing tool."""
    return f"[Mock article based on: {research[:40]}...] — Draft complete."


async def run_content_crew(topic: str) -> str:
    llm = MockLLM()
    collector = TraceCollector(model=llm.model, task="content-creation")
    writer = TraceWriter(TraceWriterConfig(output_dir=Path("./traces")))

    # Turn 1: kickoff instruction
    turn_id = collector.start_turn("user", f"Create content about: {topic}")
    collector.end_turn(turn_id, latency_ms=0)

    # Turn 2: researcher agent
    research_response = llm.invoke(topic)
    turn_id = collector.start_turn("assistant", research_response.content)
    research_result = research_topic(topic)
    collector.record_tool_call(
        turn_id,
        ToolCall(
            tool_name="research_topic",
            input_args={"topic": topic},
            output=research_result,
            success=True,
            latency_ms=55,
        ),
    )
    collector.end_turn(turn_id, latency_ms=200)

    # Turn 3: writer agent
    write_response = llm.invoke("write")
    turn_id = collector.start_turn("assistant", write_response.content)
    article = write_content(research_result)
    collector.record_tool_call(
        turn_id,
        ToolCall(
            tool_name="write_content",
            input_args={"research": research_result},
            output=article,
            success=True,
            latency_ms=70,
        ),
    )
    collector.end_turn(turn_id, latency_ms=180)

    trace = collector.finalize()
    path = writer.write(trace)
    print(f"Trace saved → {path}")
    print(f"Run ID: {trace.run_id}")
    return article


if __name__ == "__main__":
    result = asyncio.run(run_content_crew("AI agents in production"))
    print(f"\nResult: {result}")
