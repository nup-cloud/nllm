"""Schema validation — pure functions for sensor data structural checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from nllm.types import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    required_fields: tuple[str, ...]
    value_type: type = float
    valid_units: tuple[str, ...] = ()
    valid_range: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


SCHEMAS: Final[Mapping[str, SchemaSpec]] = {
    "temperature": SchemaSpec(
        required_fields=("device_id", "value", "unit"),
        valid_units=("celsius", "fahrenheit", "kelvin"),
    ),
    "humidity": SchemaSpec(
        required_fields=("device_id", "value", "unit"),
        valid_units=("percent",),
    ),
    "pressure": SchemaSpec(
        required_fields=("device_id", "value", "unit"),
        valid_units=("hpa", "mbar", "atm"),
    ),
    "gps": SchemaSpec(required_fields=("device_id", "lat", "lon")),
    "battery": SchemaSpec(
        required_fields=("device_id", "value"),
        valid_range=(0.0, 100.0),
    ),
    "motion": SchemaSpec(
        required_fields=("device_id", "detected"),
        value_type=bool,
    ),
}


def validate(data: Mapping[str, object], sensor_type: str) -> ValidationReport:
    """Validate sensor data against its schema. Pure function."""
    spec = SCHEMAS.get(sensor_type)
    if spec is None:
        return ValidationReport(False, (f"unknown_sensor_type:{sensor_type}",), ())

    errors: list[str] = []
    warnings: list[str] = []

    for field_name in spec.required_fields:
        if field_name not in data:
            errors.append(f"missing_field:{field_name}")

    if errors:
        return ValidationReport(False, tuple(errors), ())

    if "unit" in data and spec.valid_units:
        unit = str(data["unit"]).lower()
        if unit not in spec.valid_units:
            errors.append(f"invalid_unit:{data['unit']}:valid={spec.valid_units}")

    if "value" in data and spec.valid_range is not None:
        lo, hi = spec.valid_range
        try:
            val = float(data["value"])  # type: ignore[arg-type]
            if val < lo or val > hi:
                warnings.append(f"out_of_range:{val}:[{lo},{hi}]")
        except (ValueError, TypeError):
            errors.append(f"invalid_value_type:{type(data['value']).__name__}")

    return ValidationReport(
        valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
