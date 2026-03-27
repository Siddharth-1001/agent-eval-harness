# Contributing to agent-eval-harness

Thank you for your interest in contributing! This document explains how to get started and what to expect from the contribution process.

## Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup

```bash
git clone https://github.com/your-org/agent-eval-harness.git
cd agent-eval-harness
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Linting and Formatting

```bash
uv run ruff check .
uv run ruff format .
```

## How to Contribute

### Reporting Bugs

Open a GitHub issue using the **Bug Report** template. Include:
- Python version and OS
- Minimal reproducible example
- Expected vs actual behavior
- Full traceback if applicable

### Requesting Features

Open a GitHub issue using the **Feature Request** template. Describe:
- The use case and motivation
- Proposed API or behavior
- Any alternative approaches you considered

For significant changes, consider opening an RFC in the `rfcs/` directory first (see `rfcs/0000-template.md`).

### Submitting Pull Requests

1. Fork the repo and create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Write your code following the style guide below.

3. Add tests. All new code must have tests; coverage must remain at or above 85%.

4. Update `CHANGELOG.md` under `## [Unreleased]`.

5. Open a PR using the pull request template. Link the relevant issue.

6. A maintainer will review within a few business days.

## Style Guide

- Python 3.12+; use modern syntax (`X | Y` unions, `match`, etc.)
- `from __future__ import annotations` at the top of every module
- Line length: 100 characters
- All public functions and classes must have docstrings
- Type annotations on all public APIs
- Use `pydantic` for data models, `typer` for CLI commands

## Project Layout

```
agent_eval/        Source code
tests/unit/        Unit tests (no I/O, no network)
tests/integration/ Integration tests (file I/O allowed, no network)
examples/          Runnable usage examples
rfcs/              Design proposals
```

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add LangChain adapter
fix: correct latency calculation in collector
docs: expand quickstart example
test: add writer truncation edge-case tests
refactor: simplify TraceWriter._apply_truncation
```

## Release Process

Maintainers cut releases by bumping the version in `pyproject.toml`, updating `CHANGELOG.md`, and tagging `vX.Y.Z`. Releases are published to PyPI via the CI pipeline.
