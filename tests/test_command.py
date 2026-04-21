"""Tests for command bounded context — parser and prompt builder."""

import pytest
from nllm.command.parser import parse
from nllm.command.prompt import reading_to_prompt, batch_to_prompt, check_thresholds
from nllm.sensor.reading import SensorReading
from nllm.types import DeviceId


class TestParser:
    def test_single_command(self) -> None:
        r = parse("ASCEND(altitude=10, unit='meter')", "drone")
        assert r.success
        assert len(r.commands) == 1
        assert r.commands[0].action == "ASCEND"
        assert r.commands[0].params["altitude"] == 10

    def test_status_ok(self) -> None:
        r = parse("STATUS_OK")
        assert r.success
        assert len(r.commands) == 0

    def test_empty(self) -> None:
        assert not parse("").success

    def test_emergency_priority(self) -> None:
        r = parse("EMERGENCY_STOP(reason='user')", "drone")
        assert r.commands[0].priority == 2

    def test_confirmation_flag(self) -> None:
        r = parse("TAKEOFF(mode='vertical')", "drone")
        assert r.commands[0].requires_confirmation

    def test_string_param(self) -> None:
        r = parse("MOVE(direction='north', distance=100)", "drone")
        assert r.commands[0].params["direction"] == "north"
        assert r.commands[0].params["distance"] == 100


class TestPromptBuilder:
    def test_temperature(self) -> None:
        s = SensorReading(DeviceId("t1"), "temperature", 35.5, "celsius", "工場A")
        p = reading_to_prompt(s)
        assert "工場A" in p
        assert "35.5" in p

    def test_battery(self) -> None:
        s = SensorReading(DeviceId("d1"), "battery", 15, "%")
        assert "15" in reading_to_prompt(s)

    def test_batch(self) -> None:
        readings = [
            SensorReading(DeviceId("s1"), "temperature", 25, "celsius", "倉庫"),
            SensorReading(DeviceId("s2"), "humidity", 60, "percent", "工場"),
        ]
        p = batch_to_prompt(readings)
        assert "倉庫" in p
        assert "工場" in p

    def test_threshold_high_temp(self) -> None:
        s = SensorReading(DeviceId("s1"), "temperature", 45, "celsius")
        assert "temperature_high" in check_thresholds(s)

    def test_threshold_low_battery(self) -> None:
        s = SensorReading(DeviceId("s1"), "battery", 15, "%")
        assert "battery_low" in check_thresholds(s)

    def test_no_threshold(self) -> None:
        s = SensorReading(DeviceId("s1"), "temperature", 25, "celsius")
        assert check_thresholds(s) == ()
