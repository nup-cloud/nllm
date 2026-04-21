# Contributing

## Setup

```bash
git clone <repo-url> && cd nllm-oss
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -q
pytest tests/test_core.py  # single module
```

## Coding Style

- Python 3.10+. Type hints on all public functions.
- Pure functions preferred -- keep validators and sanitizers stateless.
- Use the `Result` monad (`Ok` / `Err`) from `nllm.types` instead of raising exceptions.
- No external network calls in the core runtime.

## Pull Requests

1. Create a feature branch from `main`.
2. Keep changes focused -- one concern per PR.
3. Add or update tests for any new behaviour.
4. Ensure `pytest -q` passes before submitting.
5. Write a clear PR description explaining *why*, not just *what*.
6. All PRs require review approval before merge.
7. **Only maintainers can merge.** Reviewers approve; they do not merge.

## Reporting Issues

Open a GitHub issue with reproduction steps. For security vulnerabilities,
see [SECURITY.md](SECURITY.md).
