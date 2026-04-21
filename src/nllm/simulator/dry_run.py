"""Dry-run execution — runs a TaskPlan against mock controllers."""

from __future__ import annotations

from typing import Sequence

from nllm.planning.executor import PlanExecutor, StepRecord
from nllm.planning.plan import TaskPlan
from nllm.simulator.mock_device import (
    MockCameraController,
    MockDroneController,
    MockHomeController,
    MockRobotController,
)


def dry_run(plan: TaskPlan) -> tuple[StepRecord, ...]:
    executor = PlanExecutor()
    executor.register_controller("drone", MockDroneController())
    executor.register_controller("camera", MockCameraController())
    executor.register_controller("home", MockHomeController())
    executor.register_controller("robot", MockRobotController())
    executor.set_approval(lambda _step: True)
    return executor.run(plan)


def replay(records: Sequence[StepRecord]) -> str:
    lines: list[str] = []
    for i, rec in enumerate(records, 1):
        status = "OK" if rec.success else "FAIL"
        params_str = ", ".join(f"{k}={v}" for k, v in rec.params.items()) if rec.params else ""
        line = f"[{i}] {status} {rec.action}({params_str})"
        if rec.error:
            line += f"  error={rec.error}"
        if rec.duration_ms > 0:
            line += f"  ({rec.duration_ms:.1f}ms)"
        lines.append(line)
    return "\n".join(lines)
