# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **Fixed** path traversal vulnerability in CLI `_load_trace` — run IDs now validated against UUID format
- **Fixed** XSS vulnerability in dashboard HTML — all user data escaped before DOM insertion
- **Added** Content Security Policy, X-Frame-Options, X-Content-Type-Options headers to dashboard
- **Added** run_id format validation in dashboard API endpoints
- **Added** SECURITY.md with vulnerability reporting policy and threat model

### Fixed
- **Fixed** pricing.toml opened on every trace write — now cached with `functools.cache`
- **Fixed** cost.py `_load_pricing` called on every `CostCalculator` instantiation — now cached
- **Fixed** silent error swallowing — replaced bare `except: pass` with proper logging

### Added
- Logging throughout the codebase (`agent_eval.cli`, `agent_eval.writer`, `agent_eval.cost`, `agent_eval.dashboard`)
- `py.typed` marker (PEP 561) for downstream type checking
- `.gitignore` file
- GitHub issue templates (YAML form-based) and PR template
- `docs/architecture.md` with system overview and data flow diagram
- 15+ new models in `pricing.toml` (GPT-4.5, o1/o3, Gemini 2.0/2.5, Llama 4, Mistral, DeepSeek)
- Python 3.14 to CI test matrix
- Separate lint job in CI

### Changed
- Schema `role` field now uses `Literal["user", "assistant", "system", "tool"]` instead of plain `str`
- Schema `method` field on `HallucinationFlag` now uses `Literal["schema", "semantic", "llm_judge"]`
- CI workflow scoped to `main` branch pushes/PRs with explicit `permissions: contents: read`
- README completely rewritten with architecture, security section, configuration guide, and roadmap

### Removed
- Binary `.docx` files from `docs/` directory
- Internal `plan.md` development document
- Old markdown issue templates (replaced by YAML forms)

## [0.1.0] - 2026-03-26

### Added
- Core trace engine with Pydantic v2 schema (`Trace`, `Turn`, `ToolCall`, `HallucinationFlag`)
- Thread-safe `TraceCollector` with sync and async APIs
- `TraceWriter` with configurable truncation and size limits
- `@trace_agent` decorator and `AgentTracer` context manager
- Metrics engine: tool success rate, latency (p50/p95), token cost, hallucination detection
- Three hallucination detection modes: schema, semantic, llm_judge
- Adapters: LangGraph, OpenAI Agents SDK, CrewAI, Anthropic, PydanticAI
- CLI: `run`, `list`, `show`, `compare`, `dashboard`, `version` commands
- Local dashboard (FastAPI + static HTML)
- 170+ tests, 87%+ coverage
