"""
LangChain / LangGraph example — web research agent.
Runs fully offline using MockLLM. No API key required.

# REAL USAGE:
#   from langchain_anthropic import ChatAnthropic
#   llm = ChatAnthropic(model="claude-sonnet-4-6")
#   # Then pass llm to run_research_agent() instead of MockLLM()
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig
from examples.mock_llm import MockLLM


def search_web(query: str) -> str:
    """Mock web search tool."""
    return f"[Mock results for: {query}] — Found 3 relevant articles."


async def run_research_agent(question: str) -> str:
    llm = MockLLM()
    collector = TraceCollector(model=llm.model, task="web-research")
    writer = TraceWriter(TraceWriterConfig(output_dir=Path("./traces")))

    # Turn 1: user question
    turn_id = collector.start_turn("user", question)
    collector.end_turn(turn_id, latency_ms=0)

    # Turn 2: assistant decides to search
    response = llm.invoke(question)
    turn_id = collector.start_turn("assistant", response.content)

    tool_call = ToolCall(
        tool_name="search_web",
        input_args={"query": question},
        output=search_web(question),
        success=True,
        latency_ms=42,
    )
    collector.record_tool_call(turn_id, tool_call)
    collector.end_turn(turn_id, latency_ms=150)

    # Turn 3: assistant summarizes
    summary = llm.invoke("summarize")
    turn_id = collector.start_turn("assistant", summary.content)
    collector.end_turn(turn_id, latency_ms=80)

    trace = collector.finalize()
    path = writer.write(trace)
    print(f"Trace saved → {path}")
    print(f"Run ID: {trace.run_id}")
    return summary.content


if __name__ == "__main__":
    result = asyncio.run(run_research_agent("What is LangGraph?"))
    print(f"\nResult: {result}")
