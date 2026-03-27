# RFC 0001 — Core Trace Schema

- **Status**: Accepted
- **Author(s)**: agent-eval-harness maintainers
- **Created**: 2026-03-26
- **Last Updated**: 2026-03-26
- **Supersedes**: N/A
- **Superseded by**: N/A

---

## Summary

Define the canonical JSON schema for a single agent evaluation run (a "Trace"), covering conversation turns, tool calls, token usage, latency, hallucination flags, and aggregate run statistics.

## Motivation

Interoperability between evaluation tools requires a stable, versioned trace format. Any adapter (LangChain, OpenAI Agents, CrewAI, etc.) must produce traces that conform to the same schema so metrics can be computed uniformly and traces can be replayed or visualized by a single dashboard.

Goals:
- Human-readable JSON that can be stored cheaply and inspected without tooling.
- Strongly typed via Pydantic v2 so adapters get validation for free.
- Versioned (`schema_version`) so future breaking changes can be detected.
- Rich enough to support hallucination detection, cost estimation, and SLA alerting.

## Detailed Design

### Top-level `Trace`

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `str` | `"1"` initially; bumped on breaking changes |
| `run_id` | `str` (UUID4) | Unique identifier for this run |
| `created_at` | `datetime` (UTC) | Wall-clock time at trace creation |
| `model` | `str` | Model identifier (e.g. `"claude-sonnet-4-6"`) |
| `task` | `str \| None` | Human-readable task description |
| `agent_config` | `dict` | Arbitrary agent/framework configuration |
| `turns` | `list[Turn]` | Ordered conversation turns |
| `summary` | `RunSummary \| None` | Aggregate statistics (computed post-run) |

### `Turn`

| Field | Type | Description |
|-------|------|-------------|
| `turn_id` | `int` | Zero-based index within the trace |
| `role` | `str` | `"user"` \| `"assistant"` \| `"system"` \| `"tool"` |
| `content` | `str` | Text content of the message |
| `tool_calls` | `list[ToolCall]` | Tool invocations made during this turn |
| `latency_ms` | `int` | Wall-clock latency for this turn in milliseconds |
| `tokens` | `TokenCount` | Token usage for this turn |

### `ToolCall`

| Field | Type | Description |
|-------|------|-------------|
| `call_id` | `str` (UUID4) | Unique identifier |
| `tool_name` | `str` | Name of the tool invoked |
| `input_args` | `dict` | Arguments passed to the tool |
| `output` | `Any` | Raw tool output |
| `success` | `bool` | Whether the call succeeded |
| `latency_ms` | `int` | Round-trip latency in milliseconds |
| `hallucination_flags` | `list[HallucinationFlag]` | Detected hallucinations in input args |

### `HallucinationFlag`

| Field | Type | Description |
|-------|------|-------------|
| `argument_name` | `str` | Name of the hallucinated argument |
| `expected` | `str` | Expected value or schema description |
| `received` | `Any` | Actual value received |
| `confidence` | `float` | Confidence score 0.0–1.0 |
| `method` | `str` | `"schema"` \| `"semantic"` \| `"llm_judge"` |

### `TokenCount`

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Tokens in the prompt |
| `completion_tokens` | `int` | Tokens in the completion |
| `total` | `int` (computed) | Sum of prompt + completion tokens |

### `RunSummary`

| Field | Type | Description |
|-------|------|-------------|
| `turn_count` | `int` | Total number of turns |
| `total_tool_calls` | `int` | Total tool invocations |
| `successful_tool_calls` | `int` | Successful invocations |
| `failed_tool_calls` | `int` | Failed invocations |
| `tool_success_rate` | `float` | `successful / total` (0.0 if no calls) |
| `hallucination_rate` | `float` | Fraction of tool calls with ≥1 flag |
| `total_latency_ms` | `int` | Sum of all turn latencies |
| `p50_turn_latency_ms` | `int` | Median turn latency |
| `p95_turn_latency_ms` | `int` | 95th-percentile turn latency |
| `estimated_cost_usd` | `float` | Estimated API cost |

### Serialization

- All Pydantic models use `model_dump_json(indent=2)` for disk storage.
- `datetime` fields serialize to ISO 8601 with UTC timezone.
- `Path` fields are not included in serialized output; they are configuration only.

### Versioning

The `schema_version` field starts at `"1"`. If a future change removes fields or changes field types incompatibly, the version is bumped. Readers should handle unknown versions gracefully.

## Drawbacks

- A fixed schema may not capture all framework-specific metadata. Mitigated by `agent_config` for arbitrary key/value pairs.
- Storing full tool inputs/outputs can produce large files. Mitigated by `TraceWriter` truncation settings.

## Alternatives

### Flat log format (JSONL)

A newline-delimited event log is more streaming-friendly but harder to query as a unit. The structured `Trace` document was chosen for simplicity of tooling.

### OpenTelemetry spans

OTel spans are the industry standard for distributed tracing. A future RFC may define an OTel exporter that maps `Trace` to OTel spans without replacing the core schema.

## Prior Art

- [LangSmith trace format](https://docs.smith.langchain.com/)
- [OpenAI Evals](https://github.com/openai/evals)
- [AgentBench](https://github.com/THUDM/AgentBench)

## Unresolved Questions

- Should `RunSummary` be computed automatically on `finalize()` or explicitly by the caller?
- What is the canonical cost model for `estimated_cost_usd`?
- Should `HallucinationFlag` support structured `expected` values (not just strings)?

## Implementation Plan

1. Phase 1 (this RFC): Define schema models and `TraceCollector` / `TraceWriter`.
2. Phase 2: Implement `RunSummary` computation in `metrics/`.
3. Phase 3: Add framework adapters that emit traces conforming to this schema.
4. Phase 4: OTel export bridge.
