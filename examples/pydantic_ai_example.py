"""
PydanticAI example — structured data extraction agent.
Runs fully offline using MockLLM. No API key required.

# REAL USAGE:
#   from pydantic_ai import Agent
#   from agent_eval.adapters.pydantic_ai import with_eval_harness
#   from agent_eval.tracer.writer import TraceWriterConfig
#   from pathlib import Path
#   agent = Agent("openai:gpt-4o", system_prompt="Extract structured data from text.")
#   traced = with_eval_harness(
#       agent,
#       task="data-extraction",
#       model="gpt-4o",
#       writer_config=TraceWriterConfig(output_dir=Path("./traces")),
#   )
#   result = await traced.run("Extract key facts from: ...")
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig
from examples.mock_llm import MockLLM


def extract_entities(text: str) -> dict[str, list[str]]:
    """Mock entity extraction tool."""
    return {"people": ["Alice", "Bob"], "organizations": ["Acme Corp"], "locations": ["New York"]}


async def run_extraction_agent(text: str) -> str:
    llm = MockLLM()
    collector = TraceCollector(model=llm.model, task="data-extraction")
    writer = TraceWriter(TraceWriterConfig(output_dir=Path("./traces")))

    # Turn 1: user request
    turn_id = collector.start_turn("user", f"Extract entities from: {text}")
    collector.end_turn(turn_id, latency_ms=0)

    # Turn 2: assistant extracts entities
    response = llm.invoke(text)
    turn_id = collector.start_turn("assistant", response.content)
    entities = extract_entities(text)
    collector.record_tool_call(
        turn_id,
        ToolCall(
            tool_name="extract_entities",
            input_args={"text": text},
            output=str(entities),
            success=True,
            latency_ms=38,
        ),
    )
    collector.end_turn(turn_id, latency_ms=130)

    # Turn 3: assistant formats results
    final = llm.invoke("format results")
    turn_id = collector.start_turn("assistant", final.content)
    collector.end_turn(turn_id, latency_ms=75)

    trace = collector.finalize()
    path = writer.write(trace)
    print(f"Trace saved → {path}")
    print(f"Run ID: {trace.run_id}")
    return str(entities)


if __name__ == "__main__":
    sample = "Alice and Bob met at Acme Corp headquarters in New York last Tuesday."
    result = asyncio.run(run_extraction_agent(sample))
    print(f"\nResult: {result}")
