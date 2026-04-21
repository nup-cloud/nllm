"""Tests for device bounded context — drone, camera, safety."""

import pytest
from nllm.device.command import DeviceCommand, HIGH_PRIORITY, CONFIRMATION_REQUIRED
from nllm.device.safety import SafetyPolicy, evaluate
from nllm.device.drone import DroneStatus, FlightState, execute_command
from nllm.device.camera import CameraConfig, CameraMonitor, DetectionEvent
from nllm.types import DeviceId, Severity


class TestDeviceCommand:
    def test_create_normal(self) -> None:
        cmd = DeviceCommand.create("ASCEND", {"altitude": 10}, "drone")
        assert cmd.action == "ASCEND"
        assert cmd.priority == 0
        assert cmd.requires_confirmation is False

    def test_create_emergency(self) -> None:
        cmd = DeviceCommand.create("EMERGENCY_STOP")
        assert cmd.priority == 2

    def test_create_needs_confirmation(self) -> None:
        cmd = DeviceCommand.create("TAKEOFF")
        assert cmd.requires_confirmation is True


class TestSafety:
    def test_all_clear(self) -> None:
        v = evaluate("ASCEND", {"altitude": 10}, altitude_m=5, speed_mps=3, battery_pct=80, gps_fix=True)
        assert v.approved

    def test_low_battery(self) -> None:
        v = evaluate("ASCEND", {}, altitude_m=5, speed_mps=3, battery_pct=10, gps_fix=True)
        assert not v.approved
        assert any("battery" in s for s in v.violations)

    def test_altitude_exceeded(self) -> None:
        v = evaluate("ASCEND", {"altitude": 200}, altitude_m=5, speed_mps=0, battery_pct=80, gps_fix=True)
        assert not v.approved

    def test_emergency_always_allowed(self) -> None:
        v = evaluate("EMERGENCY_STOP", {}, altitude_m=0, speed_mps=0, battery_pct=5, gps_fix=False)
        assert v.approved

    def test_land_always_allowed(self) -> None:
        v = evaluate("LAND", {}, altitude_m=0, speed_mps=0, battery_pct=1, gps_fix=False)
        assert v.approved

    def test_no_gps(self) -> None:
        v = evaluate("MOVE", {}, altitude_m=10, speed_mps=5, battery_pct=80, gps_fix=False)
        assert not v.approved


class TestDrone:
    def test_takeoff(self) -> None:
        s = DroneStatus()
        s2, rec = execute_command(s, "TAKEOFF", {"altitude": 5})
        assert rec.approved
        assert s2.state == FlightState.FLYING
        assert s2.altitude_m == 5.0

    def test_blocked_by_battery(self) -> None:
        s = DroneStatus(battery_pct=10.0)
        s2, rec = execute_command(s, "ASCEND", {"altitude": 50})
        assert not rec.approved
        assert s2 == s  # state unchanged

    def test_emergency_stop(self) -> None:
        s = DroneStatus(state=FlightState.FLYING, speed_mps=15, battery_pct=5)
        s2, rec = execute_command(s, "EMERGENCY_STOP", {})
        assert rec.approved
        assert s2.state == FlightState.EMERGENCY


class TestCamera:
    def test_register_and_list(self) -> None:
        mon = CameraMonitor()
        mon.register(CameraConfig(DeviceId("cam1"), "Front"))
        assert len(mon.cameras()) == 1

    def test_critical_auto_records(self) -> None:
        mon = CameraMonitor()
        mon.register(CameraConfig(DeviceId("cam1"), "Front"))
        event = DetectionEvent(DeviceId("cam1"), "intruder", 0.95, Severity.CRITICAL)
        mon.report(event)
        assert "cam1" in mon.status()["cameras_recording"]  # type: ignore[operator]

    def test_event_query(self) -> None:
        mon = CameraMonitor()
        mon.report(DetectionEvent(DeviceId("c1"), "motion", 0.8))
        mon.report(DetectionEvent(DeviceId("c2"), "smoke", 0.9))
        assert len(mon.events(camera_id="c1")) == 1
