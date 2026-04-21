"""Tests for alert bounded context — rules and pipeline."""

import pytest
from nllm.alert.rule import AlertRule, matches
from nllm.alert.pipeline import AlertPipeline
from nllm.types import Severity


class TestRuleMatching:
    def test_greater(self) -> None:
        r = AlertRule("x", "temp", ">", 40)
        assert matches(r, 45)
        assert not matches(r, 35)

    def test_less(self) -> None:
        r = AlertRule("x", "bat", "<", 25)
        assert matches(r, 10)
        assert not matches(r, 50)

    def test_equal(self) -> None:
        r = AlertRule("x", "x", "==", 0)
        assert matches(r, 0)
        assert not matches(r, 1)


class TestAlertPipeline:
    def test_trigger(self) -> None:
        p = AlertPipeline()
        p.add_rule(AlertRule("high_temp", "temperature", ">", 40, Severity.WARNING))
        alerts = p.process("s1", "temperature", 45)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.WARNING

    def test_no_trigger(self) -> None:
        p = AlertPipeline()
        p.add_rule(AlertRule("high_temp", "temperature", ">", 40))
        assert p.process("s1", "temperature", 25) == ()

    def test_cooldown(self) -> None:
        p = AlertPipeline()
        p.add_rule(AlertRule("x", "temperature", ">", 40, cooldown_sec=300))
        assert len(p.process("s1", "temperature", 45)) == 1
        assert len(p.process("s1", "temperature", 45)) == 0  # cooled down

    def test_acknowledge(self) -> None:
        p = AlertPipeline()
        p.add_rule(AlertRule("x", "temperature", ">", 40))
        alerts = p.process("s1", "temperature", 50)
        assert p.acknowledge(alerts[0].alert_id)
        assert p.alerts(unacknowledged_only=True) == ()

    def test_defaults(self) -> None:
        p = AlertPipeline()
        p.setup_defaults()
        assert len(p._rules) > 0

    def test_handler_called(self) -> None:
        p = AlertPipeline()
        p.add_rule(AlertRule("x", "temperature", ">", 40))
        captured: list = []
        p.on_alert(captured.append)
        p.process("s1", "temperature", 50)
        assert len(captured) == 1
