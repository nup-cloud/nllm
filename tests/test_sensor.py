"""Tests for sensor bounded context — schema, range, dedup."""

import pytest
from nllm.sensor.schema import validate, ValidationReport
from nllm.sensor.range_check import RangeChecker, RangeRule, check_range
from nllm.sensor.dedup import StreamDeduplicator


class TestSchemaValidation:
    def test_valid_temperature(self) -> None:
        r = validate({"device_id": "t1", "value": 25.5, "unit": "celsius"}, "temperature")
        assert r.valid

    def test_missing_field(self) -> None:
        r = validate({"device_id": "t1", "value": 25.5}, "temperature")
        assert not r.valid
        assert any("unit" in e for e in r.errors)

    def test_invalid_unit(self) -> None:
        r = validate({"device_id": "t1", "value": 25.5, "unit": "invalid"}, "temperature")
        assert not r.valid

    def test_unknown_sensor(self) -> None:
        r = validate({}, "nonexistent")
        assert not r.valid

    def test_battery_range_warning(self) -> None:
        r = validate({"device_id": "d1", "value": 150}, "battery")
        assert r.valid
        assert len(r.warnings) > 0

    def test_valid_motion(self) -> None:
        r = validate({"device_id": "m1", "detected": True}, "motion")
        assert r.valid


class TestRangeCheck:
    def test_ok(self) -> None:
        c = RangeChecker()
        assert c.check("temperature_celsius", 25.0).level == "ok"

    def test_critical(self) -> None:
        c = RangeChecker()
        assert c.check("temperature_celsius", 100.0).level == "critical"

    def test_warning(self) -> None:
        c = RangeChecker()
        assert c.check("temperature_celsius", 70.0).level == "warning"

    def test_custom_rule(self) -> None:
        c = RangeChecker()
        c.add_rule("custom", RangeRule("custom", 0, 100, 10, 90))
        assert c.check("custom", 50).level == "ok"
        assert c.check("custom", 95).level == "warning"
        assert c.check("custom", 110).level == "critical"

    def test_no_rule(self) -> None:
        c = RangeChecker()
        assert c.check("undefined", 42).in_range

    def test_pure_check(self) -> None:
        rule = RangeRule("x", 0, 100, 10, 90)
        assert check_range(50, rule).level == "ok"
        assert check_range(110, rule).level == "critical"


class TestDedup:
    def test_first_not_dup(self) -> None:
        d = StreamDeduplicator()
        assert not d.check({"a": 1}).is_duplicate

    def test_exact_dup(self) -> None:
        d = StreamDeduplicator()
        d.check({"a": 1})
        assert d.check({"a": 1}).is_duplicate

    def test_different_not_dup(self) -> None:
        d = StreamDeduplicator()
        d.check({"a": 1})
        assert not d.check({"a": 2}).is_duplicate

    def test_key_fields(self) -> None:
        d = StreamDeduplicator()
        d.check({"id": "s1", "v": 1, "ts": 1}, key_fields=["id", "v"])
        assert d.check({"id": "s1", "v": 1, "ts": 2}, key_fields=["id", "v"]).is_duplicate

    def test_clear(self) -> None:
        d = StreamDeduplicator()
        d.check({"a": 1})
        d.clear()
        assert d.size == 0
