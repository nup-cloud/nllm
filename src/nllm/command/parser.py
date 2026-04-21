"""Command parser — pure functions to convert LLM text into DeviceCommands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from nllm.device.command import DeviceCommand


@dataclass(frozen=True, slots=True)
class ParseResult:
    success: bool
    commands: tuple[DeviceCommand, ...] = ()
    errors: tuple[str, ...] = ()
    raw: str = ""


# ── Public API ───────────────────────────────────────────────────────

def parse(raw_output: str, domain: str = "") -> ParseResult:
    """Parse LLM text output into structured commands. Pure function."""
    if not raw_output or not raw_output.strip():
        return ParseResult(False, (), ("empty_output",), raw_output)

    text = raw_output.strip()

    if text.upper() == "STATUS_OK":
        return ParseResult(True, (), (), text)

    cmd = _parse_one(text, domain)
    if cmd is not None:
        return ParseResult(True, (cmd,), (), text)

    commands: list[DeviceCommand] = []
    errors: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        c = _parse_one(line, domain)
        if c:
            commands.append(c)
        else:
            errors.append(f"unparseable:{line[:80]}")

    if commands:
        return ParseResult(True, tuple(commands), tuple(errors), text)
    return ParseResult(False, (), tuple(errors) or ("no_valid_commands",), text)


# ── Internals ────────────────────────────────────────────────────────

_CMD_RE = re.compile(r"(\w+)\((.*)\)", re.DOTALL)


def _parse_one(text: str, domain: str) -> DeviceCommand | None:
    m = _CMD_RE.match(text)
    if m is None:
        return None
    action = m.group(1)
    params = _parse_params(m.group(2))
    return DeviceCommand.create(action, params, domain)


def _parse_params(raw: str) -> dict[str, object]:
    params: dict[str, object] = {}
    for part in _split_top_level(raw):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        params[key.strip()] = _coerce(value.strip())
    return params


def _split_top_level(s: str) -> list[str]:
    """Split on commas, respecting brackets and quotes."""
    parts: list[str] = []
    depth = 0
    in_q = False
    qch = ""
    cur = ""

    for ch in s:
        if ch in ("'", '"') and not in_q:
            in_q, qch = True, ch
            cur += ch
        elif ch == qch and in_q:
            in_q = False
            cur += ch
        elif ch in ("(", "[", "{") and not in_q:
            depth += 1
            cur += ch
        elif ch in (")", "]", "}") and not in_q:
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0 and not in_q:
            parts.append(cur)
            cur = ""
        else:
            cur += ch

    if cur.strip():
        parts.append(cur)
    return parts


def _coerce(v: str) -> object:
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    if v.startswith("["):
        try:
            return json.loads(v.replace("'", '"'))
        except json.JSONDecodeError:
            return v
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
