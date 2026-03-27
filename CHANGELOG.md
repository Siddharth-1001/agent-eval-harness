# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
