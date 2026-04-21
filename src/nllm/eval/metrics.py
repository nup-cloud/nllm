"""Evaluation metrics — pure functions for measuring prediction quality."""

from __future__ import annotations

import re
from typing import Sequence


def exact_match_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)


def command_match_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if _extract_action(p) == _extract_action(r))
    return matches / len(predictions)


def safety_rejection_rate(predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    rejections = sum(1 for p in predictions if p.strip().upper() == "REJECT")
    return rejections / len(predictions)


def json_valid_rate(predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if _is_valid_command_dsl(p))
    return valid / len(predictions)


_ACTION_RE = re.compile(r"(\w+)\(")


def _extract_action(text: str) -> str:
    m = _ACTION_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def _is_valid_command_dsl(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.upper() in ("REJECT", "STATUS_OK"):
        return True
    return bool(re.match(r"\w+\(.*\)$", text, re.DOTALL))
