"""Tests for planning bounded context — plan, executor, memory, registry."""

from nllm.planning.plan import create_plan, plan_drone_inspection, StepStatus
from nllm.planning.memory import AgentMemory
from nllm.planning.registry import ToolDef, ToolRegistry


class TestPlan:
    def test_create(self) -> None:
        p = create_plan("test", [
            {"action": "A", "domain": "drone"},
            {"action": "B", "domain": "drone"},
        ])
        assert p.plan_id.startswith("plan_")
        assert len(p.steps) == 2

    def test_next_step(self) -> None:
        p = create_plan("test", [
            {"action": "A", "domain": "drone"},
            {"action": "B", "domain": "drone"},
        ])
        assert p.next_step is not None
        assert p.next_step.action == "A"

    def test_dependency_ordering(self) -> None:
        p = create_plan("test", [
            {"action": "A", "domain": "drone"},
            {"action": "B", "domain": "drone"},
        ])
        # Manually set dependency now that we know the plan_id
        p.steps[1].depends_on = (p.steps[0].step_id,)

        step = p.next_step
        assert step is not None and step.action == "A"
        step.status = StepStatus.COMPLETED
        step2 = p.next_step
        assert step2 is not None and step2.action == "B"

    def test_drone_inspection(self) -> None:
        p = plan_drone_inspection("warehouse", 30.0)
        actions = [s.action for s in p.steps]
        assert "TAKEOFF" in actions
        assert "LAND" in actions

    def test_is_complete(self) -> None:
        p = create_plan("t", [{"action": "A", "domain": "d"}])
        assert not p.is_complete
        p.steps[0].status = StepStatus.COMPLETED
        assert p.is_complete

    def test_approval_flag(self) -> None:
        p = create_plan("t", [{"action": "TAKEOFF", "domain": "drone"}])
        assert p.steps[0].requires_approval


class TestMemory:
    def test_conversation(self) -> None:
        m = AgentMemory()
        m.add_conversation("user", "ドローンを飛ばして")
        assert len(m.recent_conversations()) == 1
        assert m.recent_conversations()[0]["role"] == "user"

    def test_command(self) -> None:
        m = AgentMemory()
        m.add_command("TAKEOFF", {"mode": "vertical"}, True)
        assert m.recent_commands()[0]["action"] == "TAKEOFF"

    def test_device_state(self) -> None:
        m = AgentMemory()
        m.update_device("d1", {"battery": 85})
        s = m.device_state("d1")
        assert s is not None and s["battery"] == 85

    def test_unknown_device(self) -> None:
        assert AgentMemory().device_state("x") is None

    def test_context_summary(self) -> None:
        m = AgentMemory()
        m.add_conversation("user", "テスト")
        assert "最近の活動" in m.context_summary()

    def test_max_history(self) -> None:
        m = AgentMemory(max_history=5)
        for i in range(10):
            m.add_conversation("user", f"msg {i}")
        assert len(m.recent_conversations()) == 5

    def test_clear(self) -> None:
        m = AgentMemory()
        m.add_conversation("user", "test")
        m.update_device("d1", {"ok": True})
        m.clear()
        assert m.recent_conversations() == ()
        assert m.device_state("d1") is None


class TestRegistry:
    def test_register_and_get(self) -> None:
        r = ToolRegistry()
        r.register(ToolDef("a", "desc", "drone", fn=lambda: None))
        assert r.get("a") is not None
        assert r.get("x") is None

    def test_by_domain(self) -> None:
        r = ToolRegistry()
        r.register(ToolDef("a", "d", "drone", fn=lambda: None))
        r.register(ToolDef("b", "d", "robot", fn=lambda: None))
        r.register(ToolDef("c", "d", "drone", fn=lambda: None))
        assert len(r.by_domain("drone")) == 2

    def test_invoke(self) -> None:
        r = ToolRegistry()
        r.register(ToolDef("add", "add", "test", fn=lambda a, b: a + b))
        assert r.invoke("add", a=1, b=2) == 3

    def test_describe(self) -> None:
        r = ToolRegistry()
        r.register(ToolDef("tool1", "A tool", "test", fn=lambda: None))
        assert "tool1" in r.describe()
