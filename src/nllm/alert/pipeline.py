"""Alert pipeline — stateful rule engine with cooldown and dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from nllm.alert.rule import AlertRule, matches
from nllm.types import Severity


@dataclass(slots=True)
class Alert:
    alert_id: str
    severity: Severity
    source: str
    message: str
    recommended_action: str = ""
    auto_execute: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


AlertHandler = Callable[[Alert], None]


class AlertPipeline:
    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._alerts: list[Alert] = []
        self._counter = 0
        self._last_fired: dict[str, float] = {}
        self._handlers: list[AlertHandler] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def on_alert(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def setup_defaults(self) -> None:
        from nllm.alert.rule import DEFAULT_RULES
        for r in DEFAULT_RULES:
            self.add_rule(r)

    def process(self, device_id: str, sensor_type: str, value: float) -> tuple[Alert, ...]:
        now = time.time()
        fired: list[Alert] = []

        for rule in self._rules:
            if rule.sensor_type != sensor_type:
                continue
            if now - self._last_fired.get(rule.name, 0) < rule.cooldown_sec:
                continue
            if not matches(rule, value):
                continue

            self._counter += 1
            alert = Alert(
                alert_id=f"alert_{self._counter:06d}",
                severity=rule.severity,
                source=device_id,
                message=f"{rule.name}: {sensor_type}={value} {rule.condition} {rule.threshold}",
                recommended_action=rule.action,
                auto_execute=rule.auto_execute,
                metadata={"sensor_type": sensor_type, "value": value, "rule": rule.name},
            )
            self._alerts.append(alert)
            self._last_fired[rule.name] = now
            fired.append(alert)

            for handler in self._handlers:
                handler(alert)

        return tuple(fired)

    def alerts(
        self,
        *,
        severity: Severity | None = None,
        unacknowledged_only: bool = False,
    ) -> tuple[Alert, ...]:
        result = self._alerts
        if severity:
            result = [a for a in result if a.severity == severity]
        if unacknowledged_only:
            result = [a for a in result if not a.acknowledged]
        return tuple(result)

    def acknowledge(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False
