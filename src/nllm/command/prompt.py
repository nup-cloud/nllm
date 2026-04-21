"""Prompt builder — pure functions to convert sensor data into LLM prompts."""

from __future__ import annotations

from typing import Final, Mapping, Sequence

from nllm.sensor.reading import SensorReading

# ── Templates ────────────────────────────────────────────────────────

_TEMPLATES: Final[Mapping[str, str]] = {
    "temperature": (
        "{location}の温度センサー（{device_id}）が{value}{unit}を計測。"
        "異常があれば対処コマンドを生成してください。"
    ),
    "humidity": (
        "{location}の湿度センサー（{device_id}）が{value}{unit}を計測。"
        "適切な対応が必要か判断してください。"
    ),
    "battery": (
        "デバイス{device_id}のバッテリー残量が{value}%。"
        "低バッテリー時のアクションを提案してください。"
    ),
    "motion": (
        "{location}の動体検知センサー（{device_id}）が反応。"
        "セキュリティ対応コマンドを生成してください。"
    ),
}

_DEFAULT_TEMPLATE: Final[str] = (
    "センサー{device_id}（{sensor_type}）: {value}{unit}。"
    "必要なアクションを提案してください。"
)


# ── Alert thresholds ─────────────────────────────────────────────────

_THRESHOLDS: Final[tuple[tuple[str, str, str, float], ...]] = (
    ("temperature_high", "temperature", ">", 40.0),
    ("temperature_low", "temperature", "<", 0.0),
    ("humidity_high", "humidity", ">", 80.0),
    ("battery_low", "battery", "<", 25.0),
    ("battery_critical", "battery", "<", 10.0),
)


# ── Public API ───────────────────────────────────────────────────────

def reading_to_prompt(reading: SensorReading) -> str:
    """Convert a single sensor reading to an LLM-ready prompt. Pure."""
    template = _TEMPLATES.get(reading.sensor_type, _DEFAULT_TEMPLATE)
    return template.format(
        device_id=reading.device_id,
        sensor_type=reading.sensor_type,
        value=reading.value,
        unit=reading.unit,
        location=reading.location or "不明",
    )


def batch_to_prompt(readings: Sequence[SensorReading]) -> str:
    """Convert multiple readings into a single summary prompt. Pure."""
    if not readings:
        return ""

    lines = ["以下のセンサーデータを分析し、異常があれば対処コマンドを生成してください：\n"]
    for r in readings:
        loc = r.location or "不明"
        lines.append(f"- {loc}/{r.device_id}: {r.sensor_type}={r.value}{r.unit}")
    lines.append(
        "\n異常値がある場合のみコマンドを生成してください。"
        "全て正常の場合は「STATUS_OK」と回答してください。"
    )
    return "\n".join(lines)


def check_thresholds(reading: SensorReading) -> tuple[str, ...]:
    """Return names of triggered thresholds. Pure."""
    triggered: list[str] = []
    for name, sensor_type, op, threshold in _THRESHOLDS:
        if reading.sensor_type != sensor_type:
            continue
        if op == ">" and reading.value > threshold:
            triggered.append(name)
        elif op == "<" and reading.value < threshold:
            triggered.append(name)
    return tuple(triggered)
