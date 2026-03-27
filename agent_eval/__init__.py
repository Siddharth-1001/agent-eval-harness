"""agent-eval-harness: Open-source evaluation framework for agentic AI systems."""

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.decorators import AgentTracer, trace_agent
from agent_eval.tracer.schema import (
    HallucinationFlag,
    RunSummary,
    TokenCount,
    ToolCall,
    Trace,
    Turn,
)
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig

__version__ = "0.1.0"

__all__ = [
    "TraceCollector",
    "AgentTracer",
    "trace_agent",
    "HallucinationFlag",
    "RunSummary",
    "Trace",
    "TokenCount",
    "ToolCall",
    "Turn",
    "TraceWriter",
    "TraceWriterConfig",
    "__version__",
]
