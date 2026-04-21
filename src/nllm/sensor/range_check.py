"""Range checker — threshold and statistical outlier detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final, Mapping


@dataclass(frozen=True, slots=True)
class RangeRule:
    sensor_type: str
    min_value: float = float("-inf")
    max_value: float = float("inf")
    warning_min: float = float("-inf")
    warning_max: float = float("inf")
    unit: str = ""


@dataclass(frozen=True, slots=True)
class CheckResult:
    value: float
    in_range: bool
    level: str  # "ok" | "warning" | "critical"
    message: str = ""


DEFAULT_RULES: Final[Mapping[str, RangeRule]] = {
    "temperature_celsius": RangeRule("temperature", -40, 85, -20, 60, "celsius"),
    "humidity_percent": RangeRule("humidity", 0, 100, 20, 80, "percent"),
    "battery_percent": RangeRule("battery", 0, 100, 20, 100, "percent"),
    "altitude_meter": RangeRule("altitude", 0, 150, 0, 120, "meter"),
    "speed_mps": RangeRule("speed", 0, 30, 0, 20, "m/s"),
    "pressure_hpa": RangeRule("pressure", 870, 1084, 950, 1050, "hPa"),
}


# ── Pure check against a single rule ────────────────────────────────

def check_range(value: float, rule: RangeRule) -> CheckResult:
    """Pure check of *value* against *rule*."""
    if value < rule.min_value or value > rule.max_value:
        return CheckResult(
            value, False, "critical",
            f"{value} outside [{rule.min_value}, {rule.max_value}]",
        )
    if value < rule.warning_min or value > rule.warning_max:
        return CheckResult(
            value, True, "warning",
            f"{value} outside warning [{rule.warning_min}, {rule.warning_max}]",
        )
    return CheckResult(value, True, "ok")


# ── Stateful checker with history ────────────────────────────────────

class RangeChecker:
    """Maintains sliding windows for statistical outlier detection."""

    def __init__(self, window_size: int = 100) -> None:
        self._rules: dict[str, RangeRule] = dict(DEFAULT_RULES)
        self._history: dict[str, deque[float]] = {}
        self._window_size = window_size

    def add_rule(self, key: str, rule: RangeRule) -> None:
        self._rules[key] = rule

    def check(self, key: str, value: float) -> CheckResult:
        self._record(key, value)
        rule = self._rules.get(key)
        if rule is None:
            return CheckResult(value, True, "ok", "no_rule")
        return check_range(value, rule)

    def check_outlier(self, key: str, value: float, z_threshold: float = 3.0) -> CheckResult:
        history = self._history.get(key)
        if history is None or len(history) < 10:
            return CheckResult(value, True, "ok", "insufficient_history")

        values = list(history)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return CheckResult(value, True, "ok")

        z = abs(value - mean) / std
        if z > z_threshold:
            return CheckResult(value, False, "warning", f"z_score={z:.2f}")
        return CheckResult(value, True, "ok")

    def _record(self, key: str, value: float) -> None:
        if key not in self._history:
            self._history[key] = deque(maxlen=self._window_size)
        self._history[key].append(value)
