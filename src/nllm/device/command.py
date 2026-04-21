"""Device command value objects — immutable representations of IoT actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

from nllm.types import CommandAction, Domain

# Commands that auto-elevate to emergency priority
HIGH_PRIORITY: Final[frozenset[str]] = frozenset({
    "EMERGENCY_STOP", "RTH", "LAND", "ALARM_ON", "SEND_ALERT",
})

# Commands requiring explicit human confirmation
CONFIRMATION_REQUIRED: Final[frozenset[str]] = frozenset({
    "TAKEOFF", "MOVE_FORWARD", "TRANSPORT", "ARM_MOVE",
    "WAYPOINT_NAV", "GOTO", "SHUTTER_CLOSE",
})


@dataclass(frozen=True, slots=True)
class DeviceCommand:
    """Immutable, validated device command."""

    action: CommandAction
    params: Mapping[str, object] = field(default_factory=dict)
    domain: str = ""
    priority: int = 0  # 0=normal, 1=high, 2=emergency
    requires_confirmation: bool = False

    @staticmethod
    def create(
        action: str,
        params: Mapping[str, object] | None = None,
        domain: str = "",
    ) -> DeviceCommand:
        priority = 2 if action in HIGH_PRIORITY else 0
        return DeviceCommand(
            action=CommandAction(action),
            params=params or {},
            domain=domain,
            priority=priority,
            requires_confirmation=action in CONFIRMATION_REQUIRED,
        )
