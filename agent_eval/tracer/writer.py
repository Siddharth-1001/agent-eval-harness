from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agent_eval.tracer.schema import Trace


class TraceWriterConfig(BaseModel):
    output_dir: Path = Path("~/.agent-eval/traces/")
    max_output_chars: int = 10_000
    max_content_chars: int = 50_000
    max_trace_size_mb: float = 5.0
    truncation_marker: str = "... [truncated, {original_len} chars total]"

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: object) -> None:
        self.output_dir = self.output_dir.expanduser()


class TraceWriter:
    """Writes Trace objects to JSON files with optional truncation."""

    def __init__(self, config: TraceWriterConfig | None = None) -> None:
        self.config = config or TraceWriterConfig()

    def _truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        marker = self.config.truncation_marker.format(original_len=len(text))
        return text[:max_chars] + marker

    def _apply_truncation(self, trace: Trace) -> Trace:
        """Return a copy of the trace with truncation applied."""
        data = trace.model_dump()
        for turn in data["turns"]:
            if isinstance(turn["content"], str):
                turn["content"] = self._truncate(turn["content"], self.config.max_content_chars)
            for call in turn["tool_calls"]:
                if isinstance(call["output"], str):
                    call["output"] = self._truncate(call["output"], self.config.max_output_chars)
        return Trace.model_validate(data)

    def write(self, trace: Trace) -> Path:
        """Write trace to disk synchronously. Returns the file path."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        truncated = self._apply_truncation(trace)
        output_path = self.config.output_dir / f"{trace.run_id}.json"
        json_str = truncated.model_dump_json(indent=2)
        # Check size limit
        size_mb = len(json_str.encode()) / (1024 * 1024)
        if size_mb > self.config.max_trace_size_mb:
            # Truncate oldest turns' content
            data = truncated.model_dump()
            for turn in data["turns"]:
                turn["content"] = self._truncate(turn["content"], 1000)
            truncated = Trace.model_validate(data)
            json_str = truncated.model_dump_json(indent=2)
        output_path.write_text(json_str, encoding="utf-8")
        return output_path

    async def async_write(self, trace: Trace) -> Path:
        """Write trace to disk asynchronously."""
        import aiofiles

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        truncated = self._apply_truncation(trace)
        output_path = self.config.output_dir / f"{trace.run_id}.json"
        json_str = truncated.model_dump_json(indent=2)
        size_mb = len(json_str.encode()) / (1024 * 1024)
        if size_mb > self.config.max_trace_size_mb:
            data = truncated.model_dump()
            for turn in data["turns"]:
                turn["content"] = self._truncate(turn["content"], 1000)
            truncated = Trace.model_validate(data)
            json_str = truncated.model_dump_json(indent=2)
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json_str)
        return output_path
