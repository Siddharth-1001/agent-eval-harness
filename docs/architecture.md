# Architecture

This document describes the internal architecture of `agent-eval-harness`.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Agent Code                         │
│  (LangChain / OpenAI / CrewAI / Anthropic / PydanticAI)    │
└───────────────┬─────────────────────────────────────────────┘
                │ uses adapter or decorator
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Tracer Layer                                               │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ TraceCollector│→ │ TraceWriter│→ │ ~/.agent-eval/   │   │
│  │ (in-memory)  │  │ (JSON file)│  │   traces/*.json   │   │
│  └──────────────┘  └────────────┘  └──────────────────┘   │
└───────────────┬─────────────────────────────────────────────┘
                │ reads traces
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Metrics Engine                                             │
│  ┌────────────┐ ┌───────────────┐ ┌─────────┐ ┌────────┐  │
│  │ToolSuccess │ │Hallucination  │ │ Latency │ │  Cost  │  │
│  │ Analyzer   │ │ Detector      │ │Analyzer │ │  Calc  │  │
│  └────────────┘ └───────────────┘ └─────────┘ └────────┘  │
└───────────────┬─────────────────────────────────────────────┘
                │ serves via API
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Presentation Layer                                         │
│  ┌────────┐  ┌───────────────────┐                         │
│  │  CLI   │  │ Dashboard (FastAPI)│                         │
│  │(Typer) │  │ + static HTML/JS  │                         │
│  └────────┘  └───────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Collection**: Your agent runs with an adapter/decorator. `TraceCollector` buffers turns and tool calls in memory.
2. **Persistence**: `TraceWriter` computes a `RunSummary`, applies truncation, and writes a JSON file to `~/.agent-eval/traces/`.
3. **Analysis**: The metrics engine reads traces and runs 4 analyzers concurrently via `asyncio.TaskGroup`.
4. **Visualization**: The CLI or dashboard presents metrics as tables, comparisons, or API responses.

## Key Modules

| Module | Responsibility |
|--------|---------------|
| `tracer/schema.py` | Pydantic v2 models (Trace, Turn, ToolCall, etc.) |
| `tracer/collector.py` | In-memory event buffer with sync/async APIs |
| `tracer/decorators.py` | `@trace_agent` decorator and `AgentTracer` context manager |
| `tracer/writer.py` | JSON persistence with truncation and size limits |
| `metrics/` | Stateless analyzers for tool success, hallucination, latency, cost |
| `adapters/` | Framework-specific integration (LangChain, OpenAI, CrewAI, Anthropic, PydanticAI) |
| `dashboard/` | FastAPI server + static HTML dashboard |
| `cli.py` | Typer CLI with `run`, `list`, `show`, `compare`, `dashboard` commands |

## Design Decisions

- **Structural typing for adapters**: The `AgentAdapter` protocol uses Python's structural typing — no inheritance required.
- **Async-first metric computation**: All analyzers are async and run concurrently via `asyncio.TaskGroup`.
- **No vendor lock-in**: No data ever leaves your machine unless you opt into LLM-as-judge mode.
- **Schema versioning**: `schema_version` field on Trace enables backward-compatible evolution.
- **Lazy framework imports**: Adapter files import framework packages inside constructors, not at module level, so installing one framework doesn't require all others.
