"""Input sanitizer — pure-function defence against prompt injection.

Every function is stateless and side-effect free.
Compose them in a pipeline for layered defence.
"""

from __future__ import annotations

import re
from typing import Final, Sequence

from nllm.types import Err, Ok, Result

# ── Banned patterns (injection indicators) ───────────────────────────

_BANNED_PATTERNS: Final[tuple[str, ...]] = (
    "システムプロンプトを無視",
    "以前の指示を無視",
    "ignore previous instructions",
    "ignore all prior",
    "disregard your instructions",
    "you are now",
    "pretend you are",
    "jailbreak",
    "DAN mode",
    "sudo",
    "rm -rf",
    "__import__",
    "exec(",
    "eval(",
    "os.system",
    "subprocess",
    "import os",
    "import sys",
    "<script>",
    "javascript:",
    "; DROP TABLE",
    "UNION SELECT",
    "安全チェックを無効",
    "ホワイトリストを無視",
    "制限を解除",
    "管理者モード",
    "override safety",
    "bypass whitelist",
    "disable safety",
    "shell command",
)

_BANNED_LOWER: Final[tuple[str, ...]] = tuple(p.lower() for p in _BANNED_PATTERNS)

_CONTROL_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_INPUT_LENGTH: Final[int] = 2000

# ── PII patterns (personal identifiable information) ─────────────────

_PII_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("email", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("phone_jp", re.compile(r"0\d{1,4}-\d{1,4}-\d{3,4}")),
    ("phone_mobile", re.compile(r"0[789]0-?\d{4}-?\d{4}")),
    ("credit_card", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("my_number", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("ip_address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("postal_code_jp", re.compile(r"\b\d{3}-\d{4}\b")),
)

_MASK = "[REDACTED]"


# ── Public API ───────────────────────────────────────────────────────

def sanitize_input(text: str) -> Result[str, str]:
    """Return cleaned text or a rejection reason.

    Pure function — no logging, no side effects.
    Applies injection detection, PII masking, and control char removal.
    """
    if not text or not text.strip():
        return Err("empty_input")

    lower = text.lower()
    for pattern in _BANNED_LOWER:
        if pattern in lower:
            return Err(f"injection_detected:{pattern}")

    cleaned = _CONTROL_CHAR_RE.sub("", text)
    cleaned = mask_pii(cleaned)

    if len(cleaned) > MAX_INPUT_LENGTH:
        cleaned = cleaned[:MAX_INPUT_LENGTH]

    return Ok(cleaned.strip())


def mask_pii(text: str) -> str:
    """Replace personally identifiable information with [REDACTED]. Pure."""
    result = text
    for _, pattern in _PII_PATTERNS:
        result = pattern.sub(_MASK, result)
    return result


def contains_pii(text: str) -> tuple[str, ...]:
    """Return names of PII types detected in text. Pure."""
    found: list[str] = []
    for name, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return tuple(found)


def validate_command_in_whitelist(
    action: str,
    domain: str,
    whitelist: dict[str, Sequence[str]],
) -> Result[str, str]:
    """Check that *action* is permitted for *domain*."""
    allowed = whitelist.get(domain)
    if allowed is None:
        return Err(f"unknown_domain:{domain}")
    if action not in allowed:
        return Err(f"blocked_command:{action}:{domain}")
    return Ok(action)
