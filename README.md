# agent-eval-harness

[![CI](https://github.com/Siddharth-1001/agent-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/agent-eval-harness/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/agent-eval-harness)](https://pypi.org/project/agent-eval-harness/)

> Lightweight, open-source evaluation harness for agentic AI systems — trace, measure, and compare your AI agents in minutes. No vendor lock-in. All data stays on your machine.

---

## Why agent-eval-harness?

Production AI agents fail in subtle ways that are invisible without structured observability:

- **Hallucinated tool arguments** — LLMs fabricate function parameters that look plausible but are wrong
- **Silent latency regressions** — a model update doubles response time and nobody notices
- **Cost creep** — token usage grows 3x after a prompt change
- **Tool failures** — success rates drop from 95% to 60% across deployments

`agent-eval-harness` gives you structured traces, automated metrics, and side-by-side comparisons — without sending data to any hosted platform.

## Quickstart

```bash
# Install
pip install agent-eval-harness

# Run an example (no API key needed — uses mock LLM)
python -m examples.langchain_example

# View results
agent-eval list
agent-eval show <run_id>

# Start the local dashboard
agent-eval dashboard
```

Open [http://127.0.0.1:7000](http://127.0.0.1:7000) to see the dashboard.

## Installation

```bash
# Core (no framework dependencies)
pip install agent-eval-harness

# With a specific framework
pip install 'agent-eval-harness[langchain]'
pip install 'agent-eval-harness[openai]'
pip install 'agent-eval-harness[anthropic]'
pip install 'agent-eval-harness[crewai]'
pip install 'agent-eval-harness[pydantic-ai]'

# All frameworks
pip install 'agent-eval-harness[all]'

# Development
pip install 'agent-eval-harness[dev]'
```

**Requirements:** Python 3.12+

## Supported Frameworks

| Framework | Install Extra | Adapter | Integration Style |
|-----------|--------------|---------|-------------------|
| **LangGraph / LangChain** | `langchain` | `LangGraphTracer` | Context manager with callback handler |
| **OpenAI Agents SDK** | `openai` | `trace_openai_agent` | Decorator with auto-injected hooks |
| **CrewAI** | `crewai` | `EvalHarnessCrew` | Wrapper class with step callback |
| **Anthropic** | `anthropic` | `TracedAnthropicClient` | Client wrapper intercepting API calls |
| **PydanticAI** | `pydantic-ai` | `with_eval_harness` | Agent wrapper extracting tool calls |

**Not using a framework?** The `@trace_agent` decorator and `AgentTracer` context manager work with any Python code.

## Usage

### Option 1: Decorator (simplest)

```python
from agent_eval import trace_agent

@trace_agent(task="web-research", model="claude-sonnet-4-6")
async def my_agent(query: str) -> str:
    # Your agent logic here
    return result
```

### Option 2: Context Manager (full control)

```python
from agent_eval import AgentTracer, ToolCall

async with AgentTracer(task="data-extraction", model="gpt-4o") as tracer:
    turn_id = await tracer.collector.async_start_turn("user", query)
    await tracer.collector.async_end_turn(turn_id)

    # Run your agent...
    result = await my_agent(query)

    asst_id = await tracer.collector.async_start_turn("assistant", str(result))
    await tracer.collector.async_record_tool_call(asst_id, ToolCall(
        tool_name="search", input_args={"q": query},
        output=str(result), success=True, latency_ms=150,
    ))
    await tracer.collector.async_end_turn(asst_id, latency_ms=200)
```

### Option 3: Framework Adapter

```python
# LangGraph
from agent_eval.adapters.langchain import LangGraphTracer

async with LangGraphTracer(task="research", model="gpt-4o") as tracer:
    result = await graph.ainvoke(state, config=tracer.langgraph_config)

# Anthropic
from agent_eval.adapters.anthropic import TracedAnthropicClient

client = TracedAnthropicClient(task="summarize")
response = client.messages.create(model="claude-sonnet-4-6", ...)

# OpenAI Agents SDK
from agent_eval.adapters.openai_agents import trace_openai_agent

@trace_openai_agent(task="planning", model="gpt-4o")
async def run(input_text: str) -> str:
    result = await Runner.run(agent, input_text)
    return result.final_output
```

## Metrics

Every trace is automatically analyzed across four dimensions:

| Metric | What It Measures | Key Output |
|--------|-----------------|------------|
| **Tool Success** | Per-tool success/failure rates | `success_rate`, per-tool breakdown |
| **Hallucination** | Fabricated or invalid tool arguments | 3 detection modes (see below) |
| **Latency** | Turn-level and tool-level timing | `p50`, `p95`, slowest turn |
| **Cost** | Token-based cost estimate | Per-turn and total USD |

### Hallucination Detection Modes

| Mode | How It Works | Confidence | Requires API? |
|------|-------------|------------|---------------|
| **Schema** | Validates `input_args` against a JSON schema | 1.0 (deterministic) | No |
| **Semantic** | Checks values against allowed sets | 0.9 | No |
| **LLM Judge** | Sends tool call to a judge LLM for evaluation | Variable | Yes |

```python
from agent_eval.metrics.hallucination import (
    HallucinationConfig, ToolHallucinationConfig, HallucinationDetector,
)

config = HallucinationConfig(tools={
    "search_web": ToolHallucinationConfig(mode="schema", schema={
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    }),
    "create_user": ToolHallucinationConfig(mode="semantic", value_sets={
        "role": ["admin", "viewer", "editor"],
    }),
})
```

## CLI Commands

```bash
agent-eval version                          # Show version
agent-eval run --task script.py             # Run a script and save its trace
agent-eval list                             # List all evaluation runs
agent-eval show <run_id>                    # Show metrics for a run
agent-eval compare <run_a> <run_b>          # Compare two runs side-by-side
agent-eval compare <a> <b> --export out.html  # Export comparison as HTML
agent-eval dashboard --port 7000            # Start local dashboard
```

## Configuration

### Trace Writer

```python
from agent_eval import TraceWriterConfig

config = TraceWriterConfig(
    output_dir="~/.agent-eval/traces/",  # Where traces are saved
    max_output_chars=10_000,              # Truncate tool outputs
    max_content_chars=50_000,             # Truncate turn content
    max_trace_size_mb=5.0,               # Max file size
)
```

### Cost Overrides

```python
from agent_eval.metrics.cost import MetricsConfig

config = MetricsConfig(pricing_overrides={
    "my-custom-model": {"input_per_1m": 1.0, "output_per_1m": 3.0},
})
```

Built-in pricing covers 25+ models including Claude, GPT, Gemini, Llama, Mistral, and DeepSeek families. See [`agent_eval/data/pricing.toml`](agent_eval/data/pricing.toml).

## Architecture

```
agent_eval/
├── tracer/          # Core trace collection and persistence
│   ├── schema.py    # Pydantic v2 models (Trace, Turn, ToolCall)
│   ├── collector.py # In-memory event buffer (sync + async)
│   ├── decorators.py# @trace_agent and AgentTracer
│   └── writer.py    # JSON file writer with truncation
├── metrics/         # Stateless analyzers (all async, run in parallel)
│   ├── tool_success.py
│   ├── hallucination.py  # 3 detection modes
│   ├── latency.py
│   └── cost.py
├── adapters/        # Framework integrations (lazy imports)
├── dashboard/       # FastAPI + static HTML dashboard
├── data/            # pricing.toml (model costs)
└── cli.py           # Typer CLI
```

See [docs/architecture.md](docs/architecture.md) for the full architecture document.

## Security

- **No data leaves your machine** unless you explicitly enable LLM-as-judge mode
- **Dashboard binds to 127.0.0.1 only** — not exposed to the network
- **Path traversal protection** on all filesystem operations
- **XSS protection** with Content Security Policy headers
- **Input validation** on all run IDs and user-supplied parameters

See [SECURITY.md](SECURITY.md) for the full security policy and how to report vulnerabilities.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup with `uv`
- Code style guide (Python 3.12+, ruff, pyright)
- Test requirements (85%+ coverage)
- Commit conventions

```bash
git clone https://github.com/your-org/agent-eval-harness.git
cd agent-eval-harness
uv sync --all-extras
uv run pytest          # 241 tests, 87%+ coverage
uv run ruff check .    # Lint
```

## Roadmap

- [ ] OpenTelemetry trace export
- [ ] Multi-run aggregation and trend charts
- [ ] Plugin system for custom metrics
- [ ] Streaming trace support
- [ ] Trace replay and debugging
- [ ] CI integration (GitHub Action for agent regression testing)

## License

[MIT](LICENSE)
