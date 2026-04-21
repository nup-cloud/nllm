"""Plan executor — runs task plans step-by-step with safety gates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from nllm.planning.plan import StepStatus, TaskPlan, TaskStep


class DeviceController(Protocol):
    def execute(self, action: str, params: dict[str, object] | None = None) -> object: ...


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_id: str
    action: str
    params: Mapping[str, object]
    success: bool
    result: Mapping[str, object] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


ApprovalFn = Callable[[TaskStep], bool]


class PlanExecutor:
    """Executes plans against registered device controllers."""

    def __init__(self) -> None:
        self._controllers: dict[str, DeviceController] = {}
        self._history: list[StepRecord] = []
        self._approval_fn: ApprovalFn | None = None

    def register_controller(self, domain: str, ctrl: DeviceController) -> None:
        self._controllers[domain] = ctrl

    def set_approval(self, fn: ApprovalFn) -> None:
        self._approval_fn = fn

    @property
    def history(self) -> tuple[StepRecord, ...]:
        return tuple(self._history)

    def run(self, plan: TaskPlan, *, abort_on_fail: bool = True) -> tuple[StepRecord, ...]:
        plan.status = StepStatus.RUNNING
        records: list[StepRecord] = []

        while True:
            step = plan.next_step
            if step is None:
                break

            rec = self._run_step(step)
            records.append(rec)

            if not rec.success and abort_on_fail:
                plan.status = StepStatus.FAILED
                for s in plan.steps:
                    if s.status == StepStatus.PENDING:
                        s.status = StepStatus.CANCELLED
                return tuple(records)

        plan.status = StepStatus.COMPLETED if plan.is_complete else StepStatus.FAILED
        return tuple(records)

    def _run_step(self, step: TaskStep) -> StepRecord:
        step.status = StepStatus.RUNNING

        if step.requires_approval:
            if self._approval_fn is None or not self._approval_fn(step):
                step.status = StepStatus.CANCELLED
                return StepRecord(step.step_id, step.action, step.params, False, error="approval_denied")

        ctrl = self._controllers.get(step.domain)
        if ctrl is None:
            step.status = StepStatus.FAILED
            return StepRecord(step.step_id, step.action, step.params, False, error=f"no_controller:{step.domain}")

        t0 = time.monotonic()
        try:
            raw = ctrl.execute(step.action, dict(step.params))
            ms = (time.monotonic() - t0) * 1000
            result = raw if isinstance(raw, dict) else {"raw": str(raw)}

            if isinstance(raw, dict) and not raw.get("approved", True):
                step.status = StepStatus.FAILED
                return StepRecord(step.step_id, step.action, step.params, False, result, raw.get("reason", ""), ms)

            step.status = StepStatus.COMPLETED
            step.result = result
            rec = StepRecord(step.step_id, step.action, step.params, True, result, duration_ms=ms)
        except Exception as exc:
            ms = (time.monotonic() - t0) * 1000
            step.status = StepStatus.FAILED
            rec = StepRecord(step.step_id, step.action, step.params, False, error=str(exc), duration_ms=ms)

        self._history.append(rec)
        return rec
