# agent-eval-harness

> Lightweight, open-source evaluation harness for agentic AI systems — trace, measure, and compare your AI agents in minutes.

## Why this exists

Production AI agents fail in subtle ways: tools get called with hallucinated arguments, latency spikes are invisible, costs creep up, and regressions hide between model updates. `agent-eval-harness` gives you structured observability without locking you into a hosted platform.

## Quickstart

```bash
pip install agent-eval-harness

# Run the example (no API key needed)
python examples/langchain_example.py

# View results
agent-eval list
agent-eval dashboard
```

## Supported Frameworks

| Framework | Install | Adapter |
|---|---|---|
| LangGraph | `pip install 'agent-eval-harness[langchain]'` | `LangGraphTracer` |
| OpenAI Agents SDK | `pip install 'agent-eval-harness[openai]'` | `trace_openai_agent` |
| CrewAI | `pip install 'agent-eval-harness[crewai]'` | `EvalHarnessCrew` |
| Anthropic | `pip install 'agent-eval-harness[anthropic]'` | `TracedAnthropicClient` |
| PydanticAI | `pip install 'agent-eval-harness[pydantic-ai]'` | `with_eval_harness` |

## Core Concepts

- **Trace** — a complete record of one agent run: every turn, every tool call, every token
- **Tool call** — a structured event with input args, output, success flag, and latency
- **Hallucination flag** — a detected mismatch between what the agent passed to a tool and what was expected
- **Run comparison** — a side-by-side diff of two traces across all metrics

## API Reference

### Decorators

```python
from agent_eval import trace_agent

@trace_agent(task="web-research", model="claude-sonnet-4-6")
def my_agent(query: str) -> str: ...
```

### Context manager

```python
from agent_eval import AgentTracer

async with AgentTracer(task="data-extraction", model="gpt-4o") as tracer:
    result = await my_agent.run(prompt)
    tracer.collector.start_turn("tool", ...)
```

### Adapters

```python
from agent_eval.adapters.anthropic import TracedAnthropicClient

client = TracedAnthropicClient(task="research")
response = client.messages.create(model="claude-sonnet-4-6", ...)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
