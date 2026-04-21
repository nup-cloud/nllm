"""ROS2 adapter — LLM command bridge for robot control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nllm.device.command import CONFIRMATION_REQUIRED

try:
    import rclpy
    from rclpy.node import Node as ROS2Node
    from std_msgs.msg import String
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    ROS2Node = object  # type: ignore[assignment,misc]


def is_ros2_available() -> bool:
    return _AVAILABLE


class LLMCommandNode(ROS2Node):
    """Subscribes to /llm/input, publishes structured commands to /llm/command.

    Commands requiring approval go to /llm/approval_request instead.
    """

    def __init__(self) -> None:
        if not _AVAILABLE:
            raise RuntimeError("rclpy not available")

        super().__init__("llm_command_node")
        self._sub = self.create_subscription(String, "/llm/input", self._on_input, 10)
        self._pub_cmd = self.create_publisher(String, "/llm/command", 10)
        self._pub_approval = self.create_publisher(String, "/llm/approval_request", 10)

    def _on_input(self, msg: Any) -> None:
        from nllm.core.sanitizer import sanitize_input
        from nllm.command.parser import parse

        result = sanitize_input(msg.data)
        if result.is_err():
            self.get_logger().warning("blocked: %s", result.error)  # type: ignore[union-attr]
            return

        parsed = parse(result.value, domain="robot")  # type: ignore[union-attr]
        if not parsed.success:
            return

        for cmd in parsed.commands:
            out = String()
            out.data = json.dumps({"action": cmd.action, "params": dict(cmd.params)}, ensure_ascii=False)

            if cmd.requires_confirmation:
                self._pub_approval.publish(out)
            else:
                self._pub_cmd.publish(out)
