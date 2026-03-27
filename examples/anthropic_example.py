"""
Anthropic tool-use example — document summarizer.
Runs fully offline using MockLLM. No API key required.

# REAL USAGE:
#   from agent_eval.adapters.anthropic import TracedAnthropicClient
#   from agent_eval.tracer.writer import TraceWriterConfig
#   from pathlib import Path
#   client = TracedAnthropicClient(
#       task="document-summarization",
#       model="claude-sonnet-4-6",
#       writer_config=TraceWriterConfig(output_dir=Path("./traces")),
#   )
#   response = client.messages.create(
#       model="claude-sonnet-4-6",
#       max_tokens=1024,
#       messages=[{"role": "user", "content": "Summarize this document: ..."}],
#   )
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig
from examples.mock_llm import MockLLM


def fetch_document(url: str) -> str:
    """Mock document fetch tool."""
    return f"[Mock document from: {url}] — Lorem ipsum content about the requested topic."


def summarize_text(text: str) -> str:
    """Mock summarization tool."""
    return f"Summary: {text[:60]}..."


async def run_summarizer_agent(url: str) -> str:
    llm = MockLLM()
    collector = TraceCollector(model=llm.model, task="document-summarization")
    writer = TraceWriter(TraceWriterConfig(output_dir=Path("./traces")))

    # Turn 1: user request
    turn_id = collector.start_turn("user", f"Please summarize the document at {url}")
    collector.end_turn(turn_id, latency_ms=0)

    # Turn 2: assistant fetches the document
    response = llm.invoke(url)
    turn_id = collector.start_turn("assistant", response.content)
    doc_content = fetch_document(url)
    collector.record_tool_call(
        turn_id,
        ToolCall(
            tool_name="fetch_document",
            input_args={"url": url},
            output=doc_content,
            success=True,
            latency_ms=60,
        ),
    )
    collector.end_turn(turn_id, latency_ms=160)

    # Turn 3: assistant summarizes
    summary_response = llm.invoke("summarize")
    turn_id = collector.start_turn("assistant", summary_response.content)
    summary = summarize_text(doc_content)
    collector.record_tool_call(
        turn_id,
        ToolCall(
            tool_name="summarize_text",
            input_args={"text": doc_content},
            output=summary,
            success=True,
            latency_ms=45,
        ),
    )
    collector.end_turn(turn_id, latency_ms=110)

    trace = collector.finalize()
    path = writer.write(trace)
    print(f"Trace saved → {path}")
    print(f"Run ID: {trace.run_id}")
    return summary


if __name__ == "__main__":
    result = asyncio.run(run_summarizer_agent("https://example.com/paper.pdf"))
    print(f"\nResult: {result}")
