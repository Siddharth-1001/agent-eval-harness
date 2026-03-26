# agent-eval-harness — End-to-End Developer Plan

> **Stack baseline:** Python 3.14.x · uv · ruff · pydantic v2 · FastAPI · React 19 · Vite  
> **Target:** OSS, local-only, publishable to GitHub as a public repo  
> **Goal:** Drop-in eval harness — developers plug it into existing agent projects with minimal friction

---

## Framework Integration Decisions

Based on current production usage data (2026):

| Framework | Why Include | Downloads/Stars |
|---|---|---|
| **LangGraph** | #1 enterprise framework — Cisco, Uber, LinkedIn, JPMorgan | 34.5M/month |
| **OpenAI Agents SDK** | Fastest growing, cleanest API, successor to Swarm | 10.3M/month |
| **CrewAI** | Most GitHub stars (44k+), dominant for multi-agent | 5.2M/month |
| **Anthropic (raw)** | Native tool-use API, Claude users | Core audience |
| **PydanticAI** | Type-safe, built by the pydantic team — perfect Python 3.14 fit | Growing fast |
| ~~AutoGen/AG2~~ | ~~Skipped — in maintenance mode, Microsoft merged into Agent Framework~~ | Maintenance only |

> **Microsoft Agent Framework** (merged AutoGen + Semantic Kernel) targets GA Q1 2026 — add as v1.5 adapter once stable.

---

## Technology Stack (All Latest, 2026)

### Core
| Concern | Tool | Why |
|---|---|---|
| Package manager | **uv** | Replaces pip/poetry — 10–100x faster, built-in lockfile |
| Python | **3.14.x** | Free-threaded mode (PEP 703), improved `annotationlib`, better `asyncio` |
| Schema / validation | **pydantic v2.x** | Powers OpenAI SDK, Anthropic SDK, LangChain internals — de facto standard |
| Linting + formatting | **ruff** | Single tool replaces flake8 + black + isort |
| Type checking | **pyright** (strict) | Faster than mypy, better Python 3.14 support |
| CLI | **typer 0.15+** | Auto-generates `--help`, async support, clean ergonomics |
| Testing | **pytest 8.x + pytest-asyncio** | Latest asyncio mode (`asyncio_mode = "auto"`) |
| Coverage | **coverage.py + pytest-cov** | Enforced in CI at 85% |
| CI/CD | **GitHub Actions** | Native OSS, free for public repos |

### Dashboard
| Concern | Tool | Why |
|---|---|---|
| Backend | **FastAPI 0.115+** | Async, lightweight, serves static React build |
| Frontend | **React 19 + Vite 6** | React 19 concurrent features, fastest dev server |
| Charts | **Recharts 2.x** | Composable, React-native charting |
| UI components | **shadcn/ui** | Headless, accessible, no external runtime |
| Styling | **Tailwind CSS 4** | Zero-config purge, CSS-first approach in v4 |

### Python 3.14-Specific Features to Use
- **Free-threaded mode** (`python3.14t`) — for async trace collection with no GIL contention
- **`annotationlib`** — lazy annotation evaluation for pydantic model performance
- **`asyncio.TaskGroup`** — structured concurrency for parallel metric analyzers
- **Improved `pathlib`** — cleaner trace file I/O
- **`tomllib` (stdlib)** — read `pyproject.toml` config natively without extra deps

---

## Repository Structure (Final Target)

```
agent-eval-harness/
├── agent_eval/                  # Core Python package
│   ├── __init__.py              # Public API surface
│   ├── tracer/
│   │   ├── __init__.py
│   │   ├── collector.py         # TraceCollector — stateful event buffer
│   │   ├── schema.py            # Pydantic v2 models for trace schema v1
│   │   ├── writer.py            # TraceWriter — JSON serialization
│   │   └── decorators.py        # @trace_agent, AgentTracer context manager
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── tool_success.py      # ToolSuccessAnalyzer
│   │   ├── hallucination.py     # HallucinationDetector (3 modes)
│   │   ├── latency.py           # LatencyAnalyzer
│   │   └── cost.py              # CostCalculator + pricing config
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py              # AdapterProtocol (structural typing)
│   │   ├── langchain.py         # LangGraph / LangChain adapter
│   │   ├── openai_agents.py     # OpenAI Agents SDK adapter
│   │   ├── crewai.py            # CrewAI adapter
│   │   ├── anthropic.py         # Anthropic raw tool-use adapter
│   │   └── pydantic_ai.py       # PydanticAI adapter
│   ├── dashboard/
│   │   ├── server.py            # FastAPI app
│   │   ├── routes.py            # REST endpoints
│   │   └── static/              # Pre-built React assets (committed to repo)
│   └── cli.py                   # Typer CLI entrypoint
├── dashboard-ui/                # React 19 source
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── examples/
│   ├── langchain_example.py
│   ├── openai_agents_example.py
│   ├── crewai_example.py
│   ├── anthropic_example.py
│   └── pydantic_ai_example.py
├── rfcs/
│   ├── 0000-template.md
│   └── 0001-core-trace-schema.md
├── docs/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── publish.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── pyproject.toml               # uv + hatch, Python 3.14+
├── uv.lock
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE                      # MIT
```

---

## Phase-by-Phase Developer Plan

---

### Phase 0 — Foundation (Week 1–2)
**Goal:** Repo is alive, CI is green, a stranger can clone and run tests in under 5 minutes.

#### 0.1 — Repository Bootstrap
```bash
# Initialize with uv (not pip, not poetry)
uv init agent-eval-harness
cd agent-eval-harness
uv python pin 3.14

# Create package structure
mkdir -p agent_eval/{tracer,metrics,adapters,dashboard}
mkdir -p tests/{unit,integration}
mkdir -p examples rfcs docs .github/{workflows,ISSUE_TEMPLATE}
```

#### 0.2 — `pyproject.toml` Setup
```toml
[project]
name = "agent-eval-harness"
version = "0.1.0"
description = "Open-source evaluation framework for agentic AI systems"
requires-python = ">=3.14"
license = { text = "MIT" }
dependencies = [
    "pydantic>=2.10",
    "typer>=0.15",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "rich>=13.9",          # CLI output formatting
    "httpx>=0.28",         # Async HTTP client for LLM cost API checks
]

[project.optional-dependencies]
langchain  = ["langchain>=0.3", "langgraph>=0.3"]
openai     = ["openai-agents>=0.1"]
crewai     = ["crewai>=0.86"]
anthropic  = ["anthropic>=0.40"]
pydantic-ai = ["pydantic-ai>=0.0.20"]
all        = ["agent-eval-harness[langchain,openai,crewai,anthropic,pydantic-ai]"]
dev        = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.8",
    "pyright>=1.1.390",
    "httpx>=0.28",
]

[project.scripts]
agent-eval = "agent_eval.cli:app"

[tool.uv]
dev-dependencies = ["agent-eval-harness[dev]"]

[tool.ruff]
target-version = "py314"
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN"]

[tool.pytest.ini_options]
asyncio_mode = "auto"          # Python 3.14 asyncio mode
testpaths = ["tests"]
addopts = "--cov=agent_eval --cov-report=term-missing --cov-fail-under=85"

[tool.pyright]
pythonVersion = "3.14"
typeCheckingMode = "strict"
```

#### 0.3 — GitHub Actions CI
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4       # uv, not pip
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest
```

#### 0.4 — Community Files
- `README.md` — project overview, 5-minute quickstart, badge strip (CI, PyPI, license)
- `CONTRIBUTING.md` — from Phase 0 doc (already written)
- `CHANGELOG.md` — start with `## [Unreleased]` header
- `LICENSE` — MIT
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/pull_request_template.md`
- `rfcs/0000-template.md` + `rfcs/0001-core-trace-schema.md` (already written)

#### 0.5 — Phase 0 Exit Criteria
- [ ] `uv sync && uv run pytest` exits 0 on a fresh clone
- [ ] CI passes on every push
- [ ] README renders correctly on GitHub
- [ ] At least 3 "good first issue" labels created in GitHub

---

### Phase 1 — Core Trace Engine (Week 3–5)
**Goal:** Any Python function that calls an LLM can be wrapped and produces a valid trace JSON.

#### 1.1 — Pydantic v2 Trace Schema

File: `agent_eval/tracer/schema.py`

```python
from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
import uuid

class HallucinationFlag(BaseModel):
    argument_name: str
    expected: str
    received: Any
    confidence: float  # 0.0–1.0
    method: str        # "schema" | "semantic" | "llm_judge"

class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    input_args: dict[str, Any]
    output: Any
    success: bool
    latency_ms: int
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)

class TokenCount(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

class Turn(BaseModel):
    turn_id: int
    role: str   # "user" | "assistant" | "system" | "tool"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: int = 0
    tokens: TokenCount = Field(default_factory=TokenCount)

class RunSummary(BaseModel):
    turn_count: int
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_success_rate: float
    hallucination_rate: float
    total_latency_ms: int
    p50_turn_latency_ms: int
    p95_turn_latency_ms: int
    estimated_cost_usd: float

class Trace(BaseModel):
    schema_version: str = "1"
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model: str
    task: str | None = None
    agent_config: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    summary: RunSummary | None = None
```

#### 1.2 — TraceCollector

File: `agent_eval/tracer/collector.py`

- Stateful buffer — accumulates `TraceEvent` objects during a run
- Thread-safe using `asyncio.Lock` (Python 3.14 free-threaded compatible)
- Emits a `Trace` when `finalize()` is called
- Supports both sync and async agent runtimes

```python
# Public interface sketch
class TraceCollector:
    def start_turn(self, role: str, content: str) -> int: ...
    def record_tool_call(self, turn_id: int, call: ToolCall) -> None: ...
    def end_turn(self, turn_id: int, latency_ms: int, tokens: TokenCount) -> None: ...
    def finalize(self) -> Trace: ...
```

#### 1.3 — Instrumentation API

File: `agent_eval/tracer/decorators.py`

```python
# Two patterns — developer chooses based on their code style

# Pattern A: Decorator (function-based agents)
@trace_agent(task="Summarize customer feedback", model="claude-sonnet-4-5")
async def my_agent(prompt: str) -> str:
    ...

# Pattern B: Context manager (block-based / framework agents)
async with AgentTracer(task="Book a flight", model="gpt-4o") as tracer:
    result = await my_existing_agent.run(prompt)
# Trace is saved automatically on context exit
```

#### 1.4 — TraceWriter

File: `agent_eval/tracer/writer.py`

- Writes `{run_id}.json` to a configurable output directory
- Default: `~/.agent-eval/traces/`
- Uses `pydantic v2` `.model_dump_json()` for serialization
- Async-safe with `aiofiles`

#### 1.5 — Tests for Phase 1
```
tests/unit/
├── test_schema.py          # Schema validation, serialization round-trips
├── test_collector.py       # Event ordering, concurrent safety
└── test_writer.py          # File I/O, JSON validity

tests/integration/
└── test_trace_e2e.py       # Full decorator → JSON file pipeline
```

#### 1.6 — Phase 1 Exit Criteria
- [ ] `@trace_agent` decorator works on both sync and async functions
- [ ] `AgentTracer` context manager works with try/except (traces saved even on agent error)
- [ ] Trace JSON validates against pydantic schema on read-back
- [ ] Unit test coverage ≥ 85% on `tracer/` module
- [ ] `uv run agent-eval --help` shows a CLI (even if empty)

---

### Phase 2 — Metrics Engine (Week 6–8)
**Goal:** Given a trace JSON, produce a full metrics report with tool success, hallucination, latency, and cost.

#### 2.1 — Design Principle: Composable Analyzers

All analyzers are **stateless functions / pure classes** — they take a `Trace` and return a typed result. They use `asyncio.TaskGroup` (Python 3.14) to run in parallel.

```python
async def compute_all_metrics(trace: Trace, config: MetricsConfig) -> MetricsReport:
    async with asyncio.TaskGroup() as tg:
        tool_task    = tg.create_task(ToolSuccessAnalyzer().analyze(trace))
        halluc_task  = tg.create_task(HallucinationDetector(config).analyze(trace))
        latency_task = tg.create_task(LatencyAnalyzer().analyze(trace))
        cost_task    = tg.create_task(CostCalculator(config).analyze(trace))
    return MetricsReport(
        tool=tool_task.result(),
        hallucination=halluc_task.result(),
        latency=latency_task.result(),
        cost=cost_task.result(),
    )
```

#### 2.2 — ToolSuccessAnalyzer

File: `agent_eval/metrics/tool_success.py`

- Parses `turn.tool_calls` across all turns
- Computes: `success_rate`, `failure_count`, per-tool breakdown
- Groups failures by `error_type` (exception class name)
- Returns `ToolMetrics` pydantic model

#### 2.3 — HallucinationDetector (3 modes, all configurable per tool)

File: `agent_eval/metrics/hallucination.py`

**Mode 1: Schema Validation (always runs)**
- Compare `input_args` against the tool's registered JSON schema
- Flag type mismatches, missing required fields, extra fields

**Mode 2: Semantic Validation (optional)**
- Developer provides a value set per argument (e.g., `valid_user_ids: list[str]`)
- Check if argument value exists in the set
- Configurable: exact match or fuzzy match via `rapidfuzz`

**Mode 3: LLM-as-Judge (optional, slowest)**
- Send tool call + context to a judge LLM with a structured rubric
- Returns confidence score 0.0–1.0
- Configurable judge model (defaults to cheapest available)
- Uses structured output (pydantic) to parse judge response

**Configuration API:**
```python
from agent_eval.metrics import HallucinationConfig, ToolHallucinationConfig

config = HallucinationConfig(
    tools={
        "search_web": ToolHallucinationConfig(
            mode="schema",                          # fast mode for this tool
        ),
        "create_user": ToolHallucinationConfig(
            mode="semantic",
            value_sets={"user_role": ["admin", "viewer", "editor"]},
        ),
        "execute_code": ToolHallucinationConfig(
            mode="llm_judge",
            judge_model="claude-haiku-4-5",
            sensitivity=0.7,
        ),
    },
    default_mode="schema",   # fallback for unspecified tools
)
```

#### 2.4 — LatencyAnalyzer

File: `agent_eval/metrics/latency.py`

- Computes: `total_ms`, `p50_ms`, `p95_ms`, `slowest_turn_id`, `tool_latency_breakdown`
- Uses `statistics.quantiles()` (stdlib, no numpy needed)

#### 2.5 — CostCalculator

File: `agent_eval/metrics/cost.py`

- Token-based cost estimation
- Pricing table loaded from `pricing.toml` (uses Python 3.14 stdlib `tomllib`)
- Ships with a built-in pricing table for all major models
- Developer can override per-run via config
- Returns: `total_usd`, `cost_per_turn`, `cost_breakdown_by_model`

```toml
# agent_eval/data/pricing.toml — bundled, updated with each release
[models."claude-sonnet-4-6"]
input_per_1m  = 3.00
output_per_1m = 15.00

[models."gpt-4o"]
input_per_1m  = 2.50
output_per_1m = 10.00
```

#### 2.6 — Tests for Phase 2
```
tests/unit/
├── test_tool_success.py
├── test_hallucination_schema.py
├── test_hallucination_semantic.py
├── test_hallucination_llm_judge.py   # mocked LLM calls
├── test_latency.py
└── test_cost.py

tests/integration/
└── test_metrics_pipeline.py          # full trace → MetricsReport
```

#### 2.7 — Phase 2 Exit Criteria
- [ ] `compute_all_metrics(trace, config)` returns a fully populated `MetricsReport`
- [ ] All three hallucination modes work independently and in combination
- [ ] LLM judge calls are mockable — CI never calls a real API
- [ ] Pricing table covers top 10 models (Claude, GPT-4o, Gemini, Llama)
- [ ] Unit test coverage ≥ 85% on `metrics/` module

---

### Phase 3 — Framework Adapters (Week 9–11)
**Goal:** Developers using the 5 target frameworks plug in `agent-eval-harness` by adding 2–3 lines to existing code.

#### 3.1 — Adapter Protocol (Structural Typing)

File: `agent_eval/adapters/base.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class AgentAdapter(Protocol):
    """
    Structural protocol — any class implementing these methods is a valid adapter.
    No inheritance required.
    """
    def extract_model(self, run_output: object) -> str: ...
    def extract_turns(self, run_output: object) -> list[Turn]: ...
    def extract_tool_calls(self, turn: object) -> list[ToolCall]: ...
```

Using `Protocol` (not ABC) means existing framework objects can be adapted without modification — true zero-friction integration.

#### 3.2 — LangGraph / LangChain Adapter

File: `agent_eval/adapters/langchain.py`

- Hooks into `LangGraph`'s `astream_events` API (v2 events format)
- Captures `on_tool_start`, `on_tool_end`, `on_llm_end` events
- Maps LangGraph's `StateGraph` state to `Turn` objects
- Zero changes to existing LangGraph code

```python
# Developer usage — before
graph = build_my_langgraph_agent()
result = await graph.ainvoke({"messages": [...]})

# Developer usage — after (2 lines added)
from agent_eval.adapters.langchain import LangGraphTracer
async with LangGraphTracer(task="my task") as tracer:
    result = await graph.ainvoke({"messages": [...]}, config=tracer.langgraph_config)
```

#### 3.3 — OpenAI Agents SDK Adapter

File: `agent_eval/adapters/openai_agents.py`

- Hooks into `openai-agents` tracing API (it ships with OpenTelemetry hooks)
- Captures `ToolCallItem` and `RunItem` events
- Maps `Runner.run()` output to trace format

```python
# Developer usage
from agent_eval.adapters.openai_agents import trace_openai_agent

@trace_openai_agent(task="process invoice")
async def run():
    return await Runner.run(agent, "Process this invoice")
```

#### 3.4 — CrewAI Adapter

File: `agent_eval/adapters/crewai.py`

- Hooks into CrewAI's `step_callback` and `task_callback`
- Captures tool streaming events (added Jan 2026)
- Maps `TaskOutput` to `Turn` objects

```python
# Developer usage
from agent_eval.adapters.crewai import EvalHarnessCrew

crew = EvalHarnessCrew(  # Thin subclass of Crew, adds tracing
    agents=[...],
    tasks=[...],
    task="process customer tickets",
)
result = crew.kickoff()
```

#### 3.5 — Anthropic Raw Adapter

File: `agent_eval/adapters/anthropic.py`

- Wraps `anthropic.Anthropic()` client
- Intercepts `messages.create()` calls with `tool_use` content blocks
- No subclassing — wraps the client object

```python
# Developer usage
from agent_eval.adapters.anthropic import TracedAnthropicClient

client = TracedAnthropicClient(task="research assistant")  # drop-in replacement
response = await client.messages.create(...)  # identical API
```

#### 3.6 — PydanticAI Adapter

File: `agent_eval/adapters/pydantic_ai.py`

- Uses PydanticAI's native `UsageLimits` and `result` inspection
- Hooks into `agent.run_sync()` / `agent.run()` via dependency injection
- Natively typed — best Python 3.14 ergonomics of all adapters

```python
# Developer usage
from agent_eval.adapters.pydantic_ai import with_eval_harness

agent = with_eval_harness(my_pydantic_agent, task="data extraction")
result = await agent.run("Extract the key facts from this document")
```

#### 3.7 — Tests for Phase 3
```
tests/unit/
├── test_adapter_langchain.py    # mocked LangGraph events
├── test_adapter_openai.py
├── test_adapter_crewai.py
├── test_adapter_anthropic.py
└── test_adapter_pydantic_ai.py

tests/integration/
├── test_e2e_langchain.py        # real local agent (no API — uses mock LLM)
└── test_e2e_anthropic.py
```

#### 3.8 — Phase 3 Exit Criteria
- [ ] Each adapter has a working example in `examples/`
- [ ] Each example produces a valid trace JSON when run
- [ ] Adapters do not modify the developer's existing agent code beyond 2–3 lines
- [ ] All adapter tests use mocked LLM calls — zero API costs in CI

---

### Phase 4 — CLI & Dashboard (Week 12–14)
**Goal:** `pip install agent-eval-harness` → `agent-eval dashboard` → everything works in a browser.

#### 4.1 — CLI Commands

File: `agent_eval/cli.py` (Typer)

```bash
# Run a registered eval task
agent-eval run --task examples/langchain_example.py --output ./traces

# List all saved runs
agent-eval list

# Show metrics for a single run
agent-eval show <run_id>

# Compare two runs (diff table in terminal)
agent-eval compare <run_id_a> <run_id_b>

# Export comparison as shareable HTML
agent-eval compare <run_id_a> <run_id_b> --export comparison.html

# Start local dashboard
agent-eval dashboard [--port 7000] [--traces-dir ./traces]
```

#### 4.2 — Dashboard Architecture

**Single process, single command.** FastAPI serves both the API and the pre-built React assets.

```
agent-eval dashboard
    │
    └── uvicorn agent_eval.dashboard.server:app
            │
            ├── GET  /api/runs              → list all traces
            ├── GET  /api/runs/{run_id}     → single trace
            ├── GET  /api/runs/{run_id}/metrics  → computed metrics
            ├── POST /api/compare           → compare two run IDs
            └── GET  /*                     → serve React SPA (index.html)
```

#### 4.3 — Dashboard UI (React 19 + Vite 6)

**Pages:**
1. **Runs List** — table of all runs with key metrics at a glance (success rate, cost, latency)
2. **Run Detail** — full trace view: turn-by-turn timeline, tool call inspector, hallucination flags
3. **Compare View** — side-by-side diff of two runs across all metrics with delta indicators

**Key components:**
- `RunsTable` — sortable, filterable by framework/model/task
- `TraceTimeline` — turn-by-turn swimlane with tool call events
- `MetricsCard` — success rate / hallucination rate / cost badges
- `ComparePanel` — grouped metric diff with ▲▼ delta values
- `HallucinationDetail` — expandable per-flag inspector

**Build process:**
```bash
# In dashboard-ui/ directory
npm run build
# Outputs to agent_eval/dashboard/static/ — committed to the repo
# Developers do NOT need Node.js installed to use the dashboard
```

#### 4.4 — Phase 4 Exit Criteria
- [ ] `agent-eval --help` shows all commands with descriptions
- [ ] `agent-eval dashboard` opens in browser and loads runs
- [ ] Run detail page shows full turn timeline
- [ ] Compare view correctly diffs two runs
- [ ] Dashboard works with 0 runs (shows empty state, not crash)
- [ ] Dashboard works with 1000+ trace files (pagination, no memory explosion)

---

### Phase 5 — Examples, Docs & Public Launch (Week 15–16)
**Goal:** A developer who finds this repo on GitHub can be running their first eval in under 10 minutes.

#### 5.1 — Example Scripts

Each example in `examples/` must:
- Run fully offline using a mock LLM (no API key needed to try it out)
- Produce a real trace JSON
- Be under 50 lines of actual code (excluding comments)
- Include a `# REAL USAGE:` comment block showing how to swap in a real model

```
examples/
├── mock_llm.py              # Shared mock LLM for all examples
├── langchain_example.py     # LangGraph web research agent
├── openai_agents_example.py # OpenAI Agents SDK task agent
├── crewai_example.py        # CrewAI 3-agent crew
├── anthropic_example.py     # Raw Anthropic tool-use loop
└── pydantic_ai_example.py   # PydanticAI typed data extraction agent
```

#### 5.2 — README Structure
```markdown
# agent-eval-harness

> One-liner description

## Why this exists        (30-second pitch)
## Quickstart             (under 10 lines — pip install, run example, open dashboard)
## Supported Frameworks   (table with install commands)
## Core Concepts          (trace, tool call, hallucination flag, run comparison)
## API Reference          (link to docs)
## Contributing           (link to CONTRIBUTING.md)
## License
```

#### 5.3 — CHANGELOG Format (Keep a Changelog style)
```markdown
## [0.1.0] - 2026-XX-XX
### Added
- Core trace engine with pydantic v2 schema
- Tool call tracking with success/failure capture
- Three-mode hallucination detection
- Latency and cost metrics
- Adapters: LangGraph, OpenAI Agents SDK, CrewAI, Anthropic, PydanticAI
- Local dashboard (FastAPI + React 19)
- CLI with run, list, show, compare, dashboard commands
```

#### 5.4 — GitHub Repo Setup Checklist
- [ ] Repository description + topics set (`ai`, `agents`, `evaluation`, `llm`, `open-source`)
- [ ] `About` section filled with one-liner + website (docs link)
- [ ] Branch protection on `main` — require CI pass + 1 review
- [ ] Labels created: `bug`, `enhancement`, `good first issue`, `help wanted`, `documentation`, `breaking change`
- [ ] Pinned issues: "Roadmap v1.0", "Good First Issues"
- [ ] `SECURITY.md` added (responsible disclosure policy)
- [ ] PyPI package registered: `agent-eval-harness`
- [ ] GitHub Actions publish workflow — auto-publishes on `vX.X.X` tag push

#### 5.5 — Phase 5 Exit Criteria
- [ ] All 5 example scripts run with `uv run python examples/<name>.py` and no errors
- [ ] README renders correctly — all code blocks are valid, no broken links
- [ ] `pip install agent-eval-harness` installs cleanly on Python 3.14
- [ ] First public release: `git tag v0.1.0 && git push --tags`

---

## Testing Strategy Summary

| Level | What it tests | Mock strategy | CI time target |
|---|---|---|---|
| Unit | Individual classes/functions | Full mocks | < 30s |
| Integration | Component pipelines | Mock LLM, real file I/O | < 90s |
| Example smoke tests | All 5 example scripts | Mock LLM | < 60s |
| **Total CI budget** | | | **< 3 minutes** |

**Key rules:**
- No real LLM API calls in CI — ever
- No internet access in tests — ever
- Async tests use `asyncio_mode = "auto"` (pytest-asyncio, Python 3.14)
- All test fixtures in `conftest.py` — no repeated setup code

---

## Public API Surface (v1.0 contract)

Everything in `agent_eval/__init__.py` is stable. Everything else is internal.

```python
# Public API — stable from v1.0 onwards
from agent_eval import (
    # Instrumentation
    trace_agent,         # decorator
    AgentTracer,         # context manager

    # Adapters
    LangGraphTracer,
    trace_openai_agent,
    EvalHarnessCrew,
    TracedAnthropicClient,
    with_eval_harness,

    # Metrics
    compute_all_metrics,
    MetricsConfig,
    HallucinationConfig,

    # Schema (for reading/writing traces programmatically)
    Trace,
    Turn,
    ToolCall,
    RunSummary,
)
```

---

## Versioning Policy

| Change type | Version bump | RFC required? |
|---|---|---|
| New adapter | minor (0.x.0) | No |
| New metric | minor (0.x.0) | No |
| Public API change | minor (0.x.0) | Yes |
| Trace schema change | major (x.0.0) | Yes |
| Bug fix | patch (0.0.x) | No |

Follows [Semantic Versioning](https://semver.org). Schema version is tracked separately inside the trace JSON.

---

## v1.5 Roadmap (Post-Launch)

- **Microsoft Agent Framework adapter** (once 1.0 GA lands)
- **Google ADK adapter**
- **Multi-agent trace support** (agent-to-agent call graphs)
- **SQLite storage backend** (plugin, not default)
- **Streaming evaluation** (real-time metrics during a run)
- **Token efficiency metric** (task completion per dollar)
- **Plugin system** — community-contributed metrics and adapters via entry points

---

## Summary Table

| Phase | Weeks | Deliverable | Exit Gate |
|---|---|---|---|
| 0 — Foundation | 1–2 | Repo, CI, community docs | Green CI on fresh clone |
| 1 — Trace Engine | 3–5 | Decorator, context manager, JSON output | Valid trace from real agent code |
| 2 — Metrics Engine | 6–8 | 4 analyzers, 3 hallucination modes | Full MetricsReport from trace |
| 3 — Adapters | 9–11 | 5 framework adapters + examples | 2-line integration per framework |
| 4 — CLI & Dashboard | 12–14 | CLI + local React dashboard | Dashboard opens, loads, compares |
| 5 — Launch | 15–16 | Docs, examples, PyPI publish | `pip install` works, tag pushed |
