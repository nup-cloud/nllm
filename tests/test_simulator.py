"""Tests for simulator — mock controllers and dry-run execution."""

from nllm.device.drone import FlightState
from nllm.planning.plan import create_plan
from nllm.simulator.dry_run import dry_run, replay
from nllm.simulator.mock_device import (
    MockCameraController,
    MockDroneController,
    MockHomeController,
    MockRobotController,
)


class TestMockDroneController:
    def test_execute_logs_history(self) -> None:
        ctrl = MockDroneController()
        ctrl.execute("TAKEOFF", {"altitude": 10})
        ctrl.execute("HOVER")
        assert len(ctrl.history) == 2
        assert ctrl.history[0].action == "TAKEOFF"

    def test_state_updates(self) -> None:
        ctrl = MockDroneController()
        assert ctrl.state.state == FlightState.IDLE
        ctrl.execute("TAKEOFF", {"altitude": 20})
        assert ctrl.state.state == FlightState.FLYING
        assert ctrl.state.altitude_m == 20.0

    def test_land(self) -> None:
        ctrl = MockDroneController()
        ctrl.execute("TAKEOFF")
        ctrl.execute("LAND")
        assert ctrl.state.state == FlightState.LANDING
        assert ctrl.state.altitude_m == 0.0

    def test_returns_approved(self) -> None:
        ctrl = MockDroneController()
        result = ctrl.execute("HOVER")
        assert result["approved"] is True


class TestMockCameraController:
    def test_recording_state(self) -> None:
        ctrl = MockCameraController()
        ctrl.execute("START_RECORD")
        assert ctrl.state["recording"] is True
        ctrl.execute("STOP_RECORD")
        assert ctrl.state["recording"] is False


class TestMockHomeController:
    def test_device_state(self) -> None:
        ctrl = MockHomeController()
        ctrl.execute("LIGHT_ON", {"target": "living_room"})
        assert ctrl.state["living_room"] == "light_on"


class TestMockRobotController:
    def test_arm_move(self) -> None:
        ctrl = MockRobotController()
        ctrl.execute("ARM_MOVE", {"x": 1.0})
        assert ctrl.state["arm_state"] == "moving"

    def test_move_forward(self) -> None:
        ctrl = MockRobotController()
        ctrl.execute("MOVE_FORWARD", {"distance": 5.0})
        assert ctrl.state["position"][0] == 5.0


class TestDryRun:
    def test_simple_plan(self) -> None:
        plan = create_plan("test dry run", [
            {"action": "TAKEOFF", "params": {"altitude": 10}, "domain": "drone"},
            {"action": "HOVER", "params": {}, "domain": "drone"},
            {"action": "LAND", "params": {}, "domain": "drone"},
        ])
        records = dry_run(plan)
        assert len(records) == 3
        assert all(r.success for r in records)

    def test_replay_output(self) -> None:
        plan = create_plan("test replay", [
            {"action": "TAKEOFF", "params": {"altitude": 5}, "domain": "drone"},
            {"action": "LAND", "params": {}, "domain": "drone"},
        ])
        records = dry_run(plan)
        text = replay(records)
        assert "TAKEOFF" in text
        assert "LAND" in text
        assert "[1]" in text
        assert "OK" in text

    def test_multi_domain(self) -> None:
        plan = create_plan("multi", [
            {"action": "TAKEOFF", "params": {}, "domain": "drone"},
            {"action": "START_RECORD", "params": {}, "domain": "camera"},
            {"action": "LIGHT_ON", "params": {"target": "room"}, "domain": "home"},
            {"action": "ARM_MOVE", "params": {}, "domain": "robot"},
        ])
        records = dry_run(plan)
        assert len(records) == 4
        assert all(r.success for r in records)
