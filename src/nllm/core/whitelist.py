"""Command whitelist — immutable per-domain allowed-command registry.

Load from YAML or construct programmatically.
The whitelist itself is a frozen value object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


@dataclass(frozen=True, slots=True)
class Whitelist:
    """Immutable mapping of domain → permitted actions."""

    commands: Mapping[str, tuple[str, ...]]

    def allows(self, domain: str, action: str) -> bool:
        allowed = self.commands.get(domain, ())
        return action in allowed

    def actions_for(self, domain: str) -> tuple[str, ...]:
        return self.commands.get(domain, ())

    def domains(self) -> tuple[str, ...]:
        return tuple(self.commands.keys())


# ── Factory ──────────────────────────────────────────────────────────

def load_whitelist(path: Path) -> Whitelist:
    """Deserialise a YAML whitelist file into an immutable Whitelist."""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, list[str]] = yaml.safe_load(f)
    commands = {domain: tuple(actions) for domain, actions in raw.items()}
    return Whitelist(commands=commands)


# ── Built-in default ─────────────────────────────────────────────────

_DEFAULTS: dict[str, tuple[str, ...]] = {
    "drone": (
        "TAKEOFF", "LAND", "ASCEND", "DESCEND", "HOVER", "MOVE",
        "ROTATE", "RTH", "WAYPOINT_NAV", "CAMERA", "QUERY",
        "SET_LIMIT", "GOTO", "EMERGENCY_STOP", "PATROL",
        "PLAN_ROUTE", "IF",
    ),
    "robot": (
        "MOVE_FORWARD", "ROTATE", "STOP", "GRIP", "ARM_MOVE",
        "SET_SPEED", "GOTO", "SLAM_UPDATE", "TRANSPORT",
        "SET_SAFETY_ZONE", "DIAGNOSTIC", "SET_MODE", "QUERY",
        "MOVE_TO", "ON_DETECT",
    ),
    "camera": (
        "START_RECORD", "STOP_RECORD", "SNAPSHOT", "PTZ",
        "SET_MODE", "PLAYBACK", "FACE_DETECT", "MOTION_DETECT",
        "ANPR", "SET_ZONE", "STATUS_CHECK", "STREAM_START",
        "ON_DETECT",
    ),
    "home": (
        "AC_ON", "AC_OFF", "AC_SET", "LIGHT_ON", "LIGHT_OFF",
        "LIGHT_DIM", "LIGHT_COLOR", "TV_ON", "TV_OFF", "TV_VOLUME",
        "CURTAIN_OPEN", "CURTAIN_CLOSE", "BATH_FILL",
        "WASHER_START", "VACUUM_START", "SCENE_ACTIVATE",
        "SCHEDULE", "SHUTTER_CLOSE",
    ),
    "sensor": (
        "LOG_SENSOR", "QUERY", "STATUS_CHECK", "VISUALIZE",
        "IF", "ON_EVENT", "SEND_ALERT", "SEND_NOTIFICATION",
    ),
}

DEFAULT_WHITELIST: Whitelist = Whitelist(commands=_DEFAULTS)
