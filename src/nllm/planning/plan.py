"""Task plan aggregate — decomposed multi-step IoT operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Final, Mapping, Sequence


@unique
class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# Actions that must pass human-in-the-loop gate
APPROVAL_ACTIONS: Final[frozenset[str]] = frozenset({
    "TAKEOFF", "LAND", "MOVE_FORWARD", "TRANSPORT",
    "ARM_MOVE", "GRIP", "EMERGENCY_STOP",
    "GOTO", "WAYPOINT_NAV",
})


@dataclass(slots=True)
class TaskStep:
    step_id: str
    action: str
    params: dict[str, object] = field(default_factory=dict)
    domain: str = ""
    depends_on: tuple[str, ...] = ()
    status: StepStatus = StepStatus.PENDING
    requires_approval: bool = False
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""


@dataclass(slots=True)
class TaskPlan:
    plan_id: str
    description: str
    steps: list[TaskStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: StepStatus = StepStatus.PENDING

    @property
    def next_step(self) -> TaskStep | None:
        completed = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if all(d in completed for d in step.depends_on):
                    return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    @property
    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


# ── Plan factory functions ───────────────────────────────────────────

_counter = 0


def create_plan(description: str, steps: Sequence[Mapping[str, object]]) -> TaskPlan:
    global _counter
    _counter += 1
    plan_id = f"plan_{_counter:04d}"

    task_steps: list[TaskStep] = []
    for i, s in enumerate(steps):
        action = str(s.get("action", ""))
        deps = s.get("depends_on", ())
        task_steps.append(TaskStep(
            step_id=f"{plan_id}_s{i+1:02d}",
            action=action,
            params=dict(s.get("params", {})),  # type: ignore[arg-type]
            domain=str(s.get("domain", "")),
            depends_on=tuple(deps) if isinstance(deps, (list, tuple)) else (),
            requires_approval=action in APPROVAL_ACTIONS,
        ))

    return TaskPlan(plan_id=plan_id, description=description, steps=task_steps)


def plan_drone_inspection(target: str, altitude: float = 50.0) -> TaskPlan:
    return create_plan(
        f"drone inspection: {target}",
        [
            {"action": "STATUS_CHECK", "params": {"target": "drone"}, "domain": "drone"},
            {"action": "TAKEOFF", "params": {"mode": "vertical"}, "domain": "drone"},
            {"action": "ASCEND", "params": {"altitude": altitude, "unit": "meter"}, "domain": "drone"},
            {"action": "PATROL", "params": {"location": target, "mode": "inspection"}, "domain": "drone"},
            {"action": "CAMERA", "params": {"action": "START_RECORD"}, "domain": "drone"},
            {"action": "RTH", "params": {"mode": "auto"}, "domain": "drone"},
            {"action": "LAND", "params": {"mode": "auto"}, "domain": "drone"},
        ],
    )
