"""Traced wrapper around the Anthropic client."""

from __future__ import annotations

import time
from typing import Any

from agent_eval.tracer.collector import TraceCollector
from agent_eval.tracer.schema import TokenCount, ToolCall
from agent_eval.tracer.writer import TraceWriter, TraceWriterConfig


class _TracedMessages:
    """Wraps anthropic.resources.Messages to intercept create() calls."""

    def __init__(
        self,
        messages: Any,
        task: str | None,
        model: str,
        writer_config: TraceWriterConfig | None,
    ) -> None:
        self._messages = messages
        self._task = task
        self._model = model
        self._writer_config = writer_config

    def create(self, **kwargs: Any) -> Any:
        """Intercept messages.create(), record turns and tool calls, write trace."""
        collector = TraceCollector(model=self._model, task=self._task)

        # Record user turn from the messages param
        messages_param = kwargs.get("messages", [])
        user_content = ""
        for msg in messages_param:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list):
                    # content blocks
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    user_content = " ".join(parts)
                break

        user_turn_id = collector.start_turn("user", user_content)
        collector.end_turn(user_turn_id)

        # Call the real messages.create()
        start = time.monotonic()
        response = self._messages.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract response content
        assistant_content = ""
        tool_calls: list[ToolCall] = []

        content_blocks = getattr(response, "content", []) or []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                assistant_content += getattr(block, "text", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=getattr(block, "id", ""),
                        tool_name=getattr(block, "name", ""),
                        input_args=dict(getattr(block, "input", {}) or {}),
                        output=None,
                        success=True,
                        latency_ms=latency_ms,
                    )
                )

        # Extract token usage
        usage = getattr(response, "usage", None)
        tokens = TokenCount(
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

        assistant_turn_id = collector.start_turn("assistant", assistant_content)
        for tc in tool_calls:
            collector.record_tool_call(assistant_turn_id, tc)
        collector.end_turn(assistant_turn_id, latency_ms=latency_ms, tokens=tokens)

        # Write trace
        trace = collector.finalize()
        writer = TraceWriter(self._writer_config)
        writer.write(trace)

        return response

    async def acreate(self, **kwargs: Any) -> Any:
        """Async version of create()."""
        collector = TraceCollector(model=self._model, task=self._task)

        messages_param = kwargs.get("messages", [])
        user_content = ""
        for msg in messages_param:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    user_content = " ".join(parts)
                break

        user_turn_id = await collector.async_start_turn("user", user_content)
        await collector.async_end_turn(user_turn_id)

        start = time.monotonic()
        response = await self._messages.acreate(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        assistant_content = ""
        tool_calls: list[ToolCall] = []

        content_blocks = getattr(response, "content", []) or []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                assistant_content += getattr(block, "text", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=getattr(block, "id", ""),
                        tool_name=getattr(block, "name", ""),
                        input_args=dict(getattr(block, "input", {}) or {}),
                        output=None,
                        success=True,
                        latency_ms=latency_ms,
                    )
                )

        usage = getattr(response, "usage", None)
        tokens = TokenCount(
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

        assistant_turn_id = await collector.async_start_turn("assistant", assistant_content)
        for tc in tool_calls:
            await collector.async_record_tool_call(assistant_turn_id, tc)
        await collector.async_end_turn(assistant_turn_id, latency_ms=latency_ms, tokens=tokens)

        trace = await collector.async_finalize()
        writer = TraceWriter(self._writer_config)
        await writer.async_write(trace)

        return response


class TracedAnthropicClient:
    """
    Thin wrapper around anthropic.Anthropic that intercepts messages.create() calls,
    records tool use content blocks, and saves a trace.

    Usage::

        client = TracedAnthropicClient(task="my-task", writer_config=config)
        response = client.messages.create(model="claude-3-5-sonnet-20241022", ...)
    """

    def __init__(
        self,
        task: str | None = None,
        model: str = "unknown",
        writer_config: TraceWriterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install anthropic: pip install 'agent-eval-harness[anthropic]'"
            ) from None

        self._client = anthropic.Anthropic(**kwargs)
        self._task = task
        self._model = model
        self._writer_config = writer_config
        self.messages = _TracedMessages(
            self._client.messages,
            task=task,
            model=model,
            writer_config=writer_config,
        )
