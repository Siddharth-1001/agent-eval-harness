"""Framework adapters for agent-eval-harness."""

from __future__ import annotations

from agent_eval.adapters.anthropic import TracedAnthropicClient
from agent_eval.adapters.base import AgentAdapter
from agent_eval.adapters.crewai import EvalHarnessCrew
from agent_eval.adapters.langchain import LangGraphTracer
from agent_eval.adapters.openai_agents import trace_openai_agent
from agent_eval.adapters.pydantic_ai import with_eval_harness

__all__ = [
    "AgentAdapter",
    "TracedAnthropicClient",
    "EvalHarnessCrew",
    "LangGraphTracer",
    "trace_openai_agent",
    "with_eval_harness",
]
