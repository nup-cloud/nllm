"""Shared kernel — domain primitives, value types, and result monad.

All domain modules import their base types from here.
No module in this file imports from any other nllm module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Generic, Mapping, NewType, Sequence, TypeAlias, TypeVar

# ── Domain primitives ────────────────────────────────────────────────

DeviceId = NewType("DeviceId", str)
CommandAction = NewType("CommandAction", str)
Celsius = NewType("Celsius", float)
Meters = NewType("Meters", float)
MetersPerSecond = NewType("MetersPerSecond", float)
Percent = NewType("Percent", float)
Latitude = NewType("Latitude", float)
Longitude = NewType("Longitude", float)


# ── Domain enums ─────────────────────────────────────────────────────

@unique
class Domain(Enum):
    DRONE = "drone"
    ROBOT = "robot"
    CAMERA = "camera"
    HOME = "home"
    SENSOR = "sensor"


@unique
class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ── Result monad ─────────────────────────────────────────────────────

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


Result: TypeAlias = Ok[T] | Err[E]


def ok(value: T) -> Ok[T]:
    return Ok(value)


def err(error: E) -> Err[E]:
    return Err(error)


# ── Immutable param dict ─────────────────────────────────────────────

Params: TypeAlias = Mapping[str, object]
