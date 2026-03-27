"""Tests for the Anthropic adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_eval.tracer.writer import TraceWriterConfig


def _make_content_block(type_: str, **kwargs: object) -> MagicMock:
    block = MagicMock()
    block.type = type_
    for k, v in kwargs.items():
        setattr(block, k, v)
    return block


def _make_usage(input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_response(content_blocks: list[MagicMock], usage: MagicMock | None = None) -> MagicMock:
    response = MagicMock()
    response.content = content_blocks
    response.usage = usage or _make_usage()
    return response


@pytest.fixture()
def writer_config(tmp_path: Path) -> TraceWriterConfig:
    return TraceWriterConfig(output_dir=tmp_path)


@pytest.fixture()
def mock_anthropic_module():
    """Mock the entire anthropic module."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_client.messages = mock_messages
    mock_module.Anthropic.return_value = mock_client
    return mock_module, mock_client, mock_messages


def _make_traced_messages(mock_messages: MagicMock, writer_config: TraceWriterConfig):
    """Helper to create _TracedMessages directly without going through TracedAnthropicClient."""
    from agent_eval.adapters.anthropic import _TracedMessages

    return _TracedMessages(
        messages=mock_messages,
        task="test-task",
        model="claude-3-5-sonnet-20241022",
        writer_config=writer_config,
    )


class TestTracedMessages:
    def test_create_records_tool_use_blocks(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """Tool_use content blocks are captured as ToolCall objects in the trace."""
        tool_block = _make_content_block(
            "tool_use",
            id="toolu_01ABC",
            name="get_weather",
            input={"city": "London"},
        )
        response = _make_response([tool_block])

        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather in London?"}],
        )

        # One trace file should have been written
        trace_files = list(tmp_path.glob("*.json"))
        assert len(trace_files) == 1

        data = json.loads(trace_files[0].read_text())
        turns = data["turns"]

        # user turn + assistant turn
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

        # Tool call captured
        tool_calls = turns[1]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "get_weather"
        assert tool_calls[0]["input_args"] == {"city": "London"}

    def test_create_records_text_response(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """Non-tool text responses are captured correctly."""
        text_block = _make_content_block("text", text="Hello, how can I help?")
        response = _make_response([text_block])

        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        trace_files = list(tmp_path.glob("*.json"))
        assert len(trace_files) == 1
        data = json.loads(trace_files[0].read_text())

        turns = data["turns"]
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hello, how can I help?"
        assert turns[1]["tool_calls"] == []

    def test_create_records_mixed_content(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """Responses with both text and tool_use blocks are handled."""
        text_block = _make_content_block("text", text="Let me check that.")
        tool_block = _make_content_block(
            "tool_use",
            id="toolu_02",
            name="search",
            input={"query": "test"},
        )
        response = _make_response([text_block, tool_block])

        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Search for test"}],
        )

        trace_files = list(tmp_path.glob("*.json"))
        data = json.loads(trace_files[0].read_text())
        turns = data["turns"]
        assistant_turn = turns[1]
        assert "Let me check that." in assistant_turn["content"]
        assert len(assistant_turn["tool_calls"]) == 1
        assert assistant_turn["tool_calls"][0]["tool_name"] == "search"

    def test_create_writes_trace_file(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """Calling create() always writes a trace file."""
        response = _make_response([])
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "ping"}],
        )

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_create_returns_original_response(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """The original response object is returned unmodified."""
        response = _make_response([])
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        result = traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "ping"}],
        )
        assert result is response

    def test_multiple_tool_calls(self, tmp_path: Path, writer_config: TraceWriterConfig):
        """Multiple tool_use blocks in one response are all captured."""
        blocks = [
            _make_content_block("tool_use", id="id1", name="tool_a", input={"x": 1}),
            _make_content_block("tool_use", id="id2", name="tool_b", input={"y": 2}),
        ]
        response = _make_response(blocks)
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _make_traced_messages(mock_messages, writer_config)
        traced.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "run tools"}],
        )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        tool_calls = data["turns"][1]["tool_calls"]
        assert len(tool_calls) == 2
        assert {tc["tool_name"] for tc in tool_calls} == {"tool_a", "tool_b"}


class TestTracedAnthropicClient:
    def test_raises_import_error_when_anthropic_missing(self):
        """ImportError with install hint when anthropic package absent."""
        with patch.dict("sys.modules", {"anthropic": None}):
            import sys

            # Remove cached module
            for key in list(sys.modules.keys()):
                if "agent_eval.adapters.anthropic" in key:
                    del sys.modules[key]

            with pytest.raises(ImportError, match="pip install"):
                from agent_eval.adapters.anthropic import TracedAnthropicClient

                TracedAnthropicClient()

    def test_construction_with_mock_anthropic(
        self, tmp_path: Path, writer_config: TraceWriterConfig
    ):
        """TracedAnthropicClient constructs successfully with mocked anthropic."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            from agent_eval.adapters.anthropic import TracedAnthropicClient

            client = TracedAnthropicClient(task="t", writer_config=writer_config)
            assert client.messages is not None
