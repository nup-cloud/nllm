"""Sensor reading value object — immutable representation of a single measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from nllm.types import DeviceId


@dataclass(frozen=True, slots=True)
class SensorReading:
    device_id: DeviceId
    sensor_type: str
    value: float
    unit: str
    location: str = ""
    timestamp: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)
