"""Mock device controllers for simulation and dry-run testing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Mapping

from nllm.device.drone import DroneStatus, FlightState


@dataclass(frozen=True, slots=True)
class ControllerRecord:
    action: str
    params: Mapping[str, object]
    timestamp: float = field(default_factory=time.time)


class MockDroneController:
    def __init__(self) -> None:
        self._history: list[ControllerRecord] = []
        self._state: DroneStatus = DroneStatus()

    @property
    def history(self) -> tuple[ControllerRecord, ...]:
        return tuple(self._history)

    @property
    def state(self) -> DroneStatus:
        return self._state

    def execute(self, action: str, params: dict[str, object] | None = None) -> dict[str, object]:
        p: dict[str, object] = params or {}
        self._history.append(ControllerRecord(action=action, params=p))
        self._state = _apply_drone(self._state, action, p)
        return {"approved": True, "action": action}


class MockCameraController:
    def __init__(self) -> None:
        self._history: list[ControllerRecord] = []
        self._recording: bool = False
        self._mode: str = "idle"

    @property
    def history(self) -> tuple[ControllerRecord, ...]:
        return tuple(self._history)

    @property
    def state(self) -> dict[str, object]:
        return {"recording": self._recording, "mode": self._mode}

    def execute(self, action: str, params: dict[str, object] | None = None) -> dict[str, object]:
        p: dict[str, object] = params or {}
        self._history.append(ControllerRecord(action=action, params=p))
        if action == "START_RECORD":
            self._recording = True
            self._mode = "recording"
        elif action == "STOP_RECORD":
            self._recording = False
            self._mode = "idle"
        elif action == "CAPTURE":
            self._mode = "capture"
        return {"approved": True, "action": action}


class MockHomeController:
    def __init__(self) -> None:
        self._history: list[ControllerRecord] = []
        self._devices: dict[str, str] = {}

    @property
    def history(self) -> tuple[ControllerRecord, ...]:
        return tuple(self._history)

    @property
    def state(self) -> dict[str, object]:
        return dict(self._devices)

    def execute(self, action: str, params: dict[str, object] | None = None) -> dict[str, object]:
        p: dict[str, object] = params or {}
        self._history.append(ControllerRecord(action=action, params=p))
        target = str(p.get("target", "unknown"))
        self._devices[target] = action.lower()
        return {"approved": True, "action": action}


class MockRobotController:
    def __init__(self) -> None:
        self._history: list[ControllerRecord] = []
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._arm_state: str = "idle"

    @property
    def history(self) -> tuple[ControllerRecord, ...]:
        return tuple(self._history)

    @property
    def state(self) -> dict[str, object]:
        return {"position": self._position, "arm_state": self._arm_state}

    def execute(self, action: str, params: dict[str, object] | None = None) -> dict[str, object]:
        p: dict[str, object] = params or {}
        self._history.append(ControllerRecord(action=action, params=p))
        if action == "ARM_MOVE":
            self._arm_state = "moving"
        elif action == "GRIP":
            self._arm_state = "gripping"
        elif action == "MOVE_FORWARD":
            dx = float(p.get("distance", 1.0))
            self._position = (self._position[0] + dx, self._position[1], self._position[2])
        return {"approved": True, "action": action}


def _apply_drone(status: DroneStatus, action: str, params: Mapping[str, object]) -> DroneStatus:
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
