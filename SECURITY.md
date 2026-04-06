# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities by emailing **siddharthbhalsod147@gmail.com**.

You should receive an acknowledgment within **48 hours**. We will work with you to understand the scope, confirm the fix, and coordinate disclosure.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (optional)

## Security Design Principles

1. **No network access by default.** Traces are stored locally. The dashboard binds to `127.0.0.1` only.
2. **No secrets in traces.** The framework does not capture API keys, auth tokens, or credentials. Adapter code only records tool names, inputs, outputs, and timing.
3. **Input validation.** Run IDs are validated against UUID format before filesystem operations. All user-supplied strings rendered in HTML are escaped.
4. **Path traversal protection.** File operations sanitize user input to prevent directory traversal attacks.
5. **CSP headers.** The dashboard sets `Content-Security-Policy`, `X-Frame-Options`, and `X-Content-Type-Options` headers.
6. **No eval/exec.** The codebase never uses `eval()`, `exec()`, or `pickle.loads()` on untrusted data.

## Threat Model

### In Scope
- Path traversal via run_id parameters
- XSS in the local dashboard
- Sensitive data leakage via trace files
- Dependency vulnerabilities

### Out of Scope
- The `agent-eval run` command intentionally executes user-provided Python scripts. This is by design and runs with the user's own permissions.
- LLM judge mode sends tool call data to external APIs (Anthropic/OpenAI). Users opt into this explicitly.

## Dependencies

We regularly audit dependencies. To check for known vulnerabilities:

```bash
pip audit
```
