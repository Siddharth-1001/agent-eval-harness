"""Smoke tests: verify all example scripts can be imported and run without errors."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer_mock(tmp_path: Path):
    """Return a mock TraceWriter whose write() returns a deterministic path."""
    mock_writer = MagicMock()
    mock_writer.write.return_value = tmp_path / "fake-run-id.json"
    return mock_writer


# ---------------------------------------------------------------------------
# LangChain example
# ---------------------------------------------------------------------------


class TestLangchainExample:
    def test_import(self):
        import examples.langchain_example  # noqa: F401

    def test_run(self, tmp_path):
        import asyncio

        from examples import langchain_example

        writer_mock = _make_writer_mock(tmp_path)
        with patch("examples.langchain_example.TraceWriter", return_value=writer_mock):
            result = asyncio.run(langchain_example.run_research_agent("What is LangGraph?"))

        assert isinstance(result, str)
        assert len(result) > 0
        writer_mock.write.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI Agents example
# ---------------------------------------------------------------------------


class TestOpenAIAgentsExample:
    def test_import(self):
        import examples.openai_agents_example  # noqa: F401

    def test_run(self, tmp_path):
        import asyncio

        from examples import openai_agents_example

        writer_mock = _make_writer_mock(tmp_path)
        with patch("examples.openai_agents_example.TraceWriter", return_value=writer_mock):
            result = asyncio.run(openai_agents_example.run_planning_agent("Build a REST API"))

        assert isinstance(result, str)
        assert len(result) > 0
        writer_mock.write.assert_called_once()


# ---------------------------------------------------------------------------
# CrewAI example
# ---------------------------------------------------------------------------


class TestCrewAIExample:
    def test_import(self):
        import examples.crewai_example  # noqa: F401

    def test_run(self, tmp_path):
        import asyncio

        from examples import crewai_example

        writer_mock = _make_writer_mock(tmp_path)
        with patch("examples.crewai_example.TraceWriter", return_value=writer_mock):
            result = asyncio.run(crewai_example.run_content_crew("AI agents in production"))

        assert isinstance(result, str)
        assert len(result) > 0
        writer_mock.write.assert_called_once()


# ---------------------------------------------------------------------------
# Anthropic example
# ---------------------------------------------------------------------------


class TestAnthropicExample:
    def test_import(self):
        import examples.anthropic_example  # noqa: F401

    def test_run(self, tmp_path):
        import asyncio

        from examples import anthropic_example

        writer_mock = _make_writer_mock(tmp_path)
        with patch("examples.anthropic_example.TraceWriter", return_value=writer_mock):
            result = asyncio.run(
                anthropic_example.run_summarizer_agent("https://example.com/paper.pdf")
            )

        assert isinstance(result, str)
        assert len(result) > 0
        writer_mock.write.assert_called_once()


# ---------------------------------------------------------------------------
# PydanticAI example
# ---------------------------------------------------------------------------


class TestPydanticAIExample:
    def test_import(self):
        import examples.pydantic_ai_example  # noqa: F401

    def test_run(self, tmp_path):
        import asyncio

        from examples import pydantic_ai_example

        writer_mock = _make_writer_mock(tmp_path)
        with patch("examples.pydantic_ai_example.TraceWriter", return_value=writer_mock):
            sample = "Alice and Bob met at Acme Corp in New York."
            result = asyncio.run(pydantic_ai_example.run_extraction_agent(sample))

        assert isinstance(result, str)
        assert len(result) > 0
        writer_mock.write.assert_called_once()


# ---------------------------------------------------------------------------
# MockLLM unit tests
# ---------------------------------------------------------------------------


class TestMockLLM:
    def test_invoke_cycles(self):
        from examples.mock_llm import MockLLM

        llm = MockLLM(responses=["A", "B", "C"])
        assert llm.invoke("x").content == "A"
        assert llm.invoke("x").content == "B"
        assert llm.invoke("x").content == "C"
        assert llm.invoke("x").content == "A"  # cycles

    def test_default_responses(self):
        from examples.mock_llm import MockLLM

        llm = MockLLM()
        r = llm.invoke("hello")
        assert isinstance(r.content, str)
        assert r.model == "mock-llm-v1"

    def test_ainvoke(self):
        import asyncio

        from examples.mock_llm import MockLLM

        llm = MockLLM(responses=["async-response"])
        result = asyncio.run(llm.ainvoke("test"))
        assert result.content == "async-response"
