"""Camera aggregate — surveillance monitoring and event management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from nllm.types import DeviceId, Severity


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: DeviceId
    name: str
    location: str = ""
    resolution: str = "1080p"
    night_vision: bool = True
    motion_sensitivity: float = 0.5


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    camera_id: DeviceId
    event_type: str
    confidence: float
    severity: Severity = Severity.INFO
    metadata: Mapping[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CameraMonitor:
    """Stateful aggregate — manages camera fleet, events, and recordings."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraConfig] = {}
        self._events: list[DetectionEvent] = []
        self._recording: set[str] = set()
        self._handlers: list[Callable[[DetectionEvent], None]] = []

    def register(self, config: CameraConfig) -> None:
        self._cameras[config.camera_id] = config

    def cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(self._cameras.values())

    def on_alert(self, handler: Callable[[DetectionEvent], None]) -> None:
        self._handlers.append(handler)

    def report(self, event: DetectionEvent) -> None:
        self._events.append(event)
        if event.severity == Severity.CRITICAL:
            self._recording.add(event.camera_id)
        for handler in self._handlers:
            handler(event)

    def start_recording(self, camera_id: str) -> bool:
        if camera_id not in self._cameras:
            return False
        self._recording.add(camera_id)
        return True

    def stop_recording(self, camera_id: str) -> bool:
        if camera_id not in self._recording:
            return False
        self._recording.discard(camera_id)
        return True

    def events(
        self,
        *,
        camera_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
    ) -> tuple[DetectionEvent, ...]:
        result = self._events
        if camera_id:
            result = [e for e in result if e.camera_id == camera_id]
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if since:
            result = [e for e in result if e.timestamp >= since]
        return tuple(result)

    def status(self) -> Mapping[str, object]:
        return {
            "cameras_total": len(self._cameras),
            "cameras_recording": sorted(self._recording),
            "events_total": len(self._events),
            "events_critical": sum(
                1 for e in self._events if e.severity == Severity.CRITICAL
            ),
        }
