"""Alert rules — immutable trigger definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from nllm.types import Severity


@dataclass(frozen=True, slots=True)
class AlertRule:
    name: str
    sensor_type: str
    condition: str  # ">" | "<" | ">=" | "<=" | "=="
    threshold: float
    severity: Severity = Severity.WARNING
    action: str = ""
    auto_execute: bool = False
    cooldown_sec: float = 300.0


def matches(rule: AlertRule, value: float) -> bool:
    """Pure predicate — does *value* trigger *rule*?"""
    match rule.condition:
        case ">":
            return value > rule.threshold
        case "<":
            return value < rule.threshold
        case ">=":
            return value >= rule.threshold
        case "<=":
            return value <= rule.threshold
        case "==":
            return value == rule.threshold
        case _:
            return False


# ── Sensible defaults ────────────────────────────────────────────────

DEFAULT_RULES: Final[tuple[AlertRule, ...]] = (
    AlertRule("high_temperature", "temperature", ">", 40, Severity.WARNING, "AC_ON(mode='cool')"),
    AlertRule("low_battery", "battery", "<", 25, Severity.WARNING, "RTH(mode='auto')"),
    AlertRule("critical_battery", "battery", "<", 10, Severity.EMERGENCY, "LAND(mode='emergency')", auto_execute=True),
    AlertRule("high_altitude", "altitude", ">", 120, Severity.CRITICAL, "DESCEND(altitude=100, unit='meter')"),
)
