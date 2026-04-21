"""Drone aggregate — state machine with safety-gated command execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum, unique
from typing import Mapping

from nllm.device.safety import SafetyPolicy, SafetyVerdict, evaluate


@unique
class FlightState(Enum):
    IDLE = "idle"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDING = "landing"
    EMERGENCY = "emergency"
    RETURNING = "returning"


@dataclass(frozen=True, slots=True)
class DroneStatus:
    state: FlightState = FlightState.IDLE
    altitude_m: float = 0.0
    speed_mps: float = 0.0
    battery_pct: float = 100.0
    lat: float = 0.0
    lon: float = 0.0
    heading_deg: float = 0.0
    gps_fix: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    action: str
    params: Mapping[str, object]
    approved: bool
    reason: str = ""
    timestamp: float = 0.0


def execute_command(
    status: DroneStatus,
    action: str,
    params: Mapping[str, object],
    policy: SafetyPolicy = SafetyPolicy(),
) -> tuple[DroneStatus, ExecutionRecord]:
    """Execute a drone command, returning the new status and an audit record.

    Pure function — takes current state, returns next state.
    """
    verdict = evaluate(
        action,
        params,
        altitude_m=status.altitude_m,
        speed_mps=status.speed_mps,
        battery_pct=status.battery_pct,
        gps_fix=status.gps_fix,
        policy=policy,
    )

    if not verdict.approved:
        record = ExecutionRecord(
            action=action,
            params=params,
            approved=False,
            reason="; ".join(verdict.violations),
            timestamp=time.time(),
        )
        return status, record

    next_status = _apply(status, action, params)
    record = ExecutionRecord(
        action=action,
        params=params,
        approved=True,
        reason="ok",
        timestamp=time.time(),
    )
    return next_status, record


def _apply(
    status: DroneStatus,
    action: str,
    params: Mapping[str, object],
) -> DroneStatus:
    """Apply a command to produce a new DroneStatus. Pure transition."""
    match action:
        case "TAKEOFF":
            return replace(status, state=FlightState.FLYING, altitude_m=float(params.get("altitude", 5.0)))
        case "LAND":
            return replace(status, state=FlightState.LANDING, altitude_m=0.0, speed_mps=0.0)
        case "ASCEND":
            return replace(status, altitude_m=float(params.get("altitude", status.altitude_m + 10)))
        case "DESCEND":
            return replace(status, altitude_m=max(0.0, float(params.get("altitude", status.altitude_m - 10))))
        case "HOVER":
            return replace(status, state=FlightState.HOVERING, speed_mps=0.0)
        case "RTH":
            return replace(status, state=FlightState.RETURNING)
        case "EMERGENCY_STOP":
            return replace(status, state=FlightState.EMERGENCY, speed_mps=0.0)
        case _:
            return status
