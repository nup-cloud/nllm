"""Safety policy evaluation — pure functions for policy checks.

All functions are stateless. Pass the current state in, get a verdict out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from nllm.types import Meters, MetersPerSecond, Percent


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    max_altitude_m: Meters = Meters(150.0)
    max_speed_mps: MetersPerSecond = MetersPerSecond(20.0)
    min_battery_pct: Percent = Percent(25.0)
    max_range_m: Meters = Meters(5000.0)
    require_gps: bool = True


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    approved: bool
    violations: tuple[str, ...]


# ── Always-allowed bypass set ────────────────────────────────────────

_ALWAYS_ALLOWED: frozenset[str] = frozenset({"LAND", "EMERGENCY_STOP"})


# ── Pure evaluation ──────────────────────────────────────────────────

def evaluate(
    action: str,
    params: Mapping[str, object],
    *,
    altitude_m: float,
    speed_mps: float,
    battery_pct: float,
    gps_fix: bool,
    policy: SafetyPolicy = SafetyPolicy(),
) -> SafetyVerdict:
    """Evaluate a command against a safety policy. Pure, no side effects."""
    if action in _ALWAYS_ALLOWED:
        return SafetyVerdict(approved=True, violations=())

    violations: list[str] = []

    if battery_pct < policy.min_battery_pct:
        violations.append(
            f"battery {battery_pct}% < minimum {policy.min_battery_pct}%"
        )

    target_alt = params.get("altitude", altitude_m)
    if isinstance(target_alt, (int, float)) and target_alt > policy.max_altitude_m:
        violations.append(
            f"altitude {target_alt}m > limit {policy.max_altitude_m}m"
        )

    target_speed = params.get("speed", speed_mps)
    if isinstance(target_speed, (int, float)) and target_speed > policy.max_speed_mps:
        violations.append(
            f"speed {target_speed}m/s > limit {policy.max_speed_mps}m/s"
        )

    if policy.require_gps and not gps_fix:
        violations.append("GPS fix required but unavailable")

    return SafetyVerdict(
        approved=len(violations) == 0,
        violations=tuple(violations),
    )
