"""MQTT adapter — sensor data bridge for IoT devices."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nllm.sensor.reading import SensorReading
from nllm.types import DeviceId

try:
    import paho.mqtt.client as mqtt
    _AVAILABLE = True
except ImportError:
    mqtt = None  # type: ignore[assignment]
    _AVAILABLE = False


@dataclass(frozen=True, slots=True)
class MQTTConfig:
    broker: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = "nllm"
    sensor_topics: tuple[str, ...] = ("sensors/#",)
    command_topic_tpl: str = "commands/{device_id}"


ReadingHandler = Callable[[SensorReading], None]


class MQTTBridge:
    """Bridges MQTT sensor messages to SensorReading value objects."""

    def __init__(self, config: MQTTConfig = MQTTConfig()) -> None:
        if not _AVAILABLE:
            raise RuntimeError("paho-mqtt not installed")

        self._cfg = config
        self._client = mqtt.Client(client_id=config.client_id)
        self._handlers: list[ReadingHandler] = []

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        if config.username:
            self._client.username_pw_set(config.username, config.password)

    def on_reading(self, handler: ReadingHandler) -> None:
        self._handlers.append(handler)

    def start(self) -> None:
        self._client.connect(self._cfg.broker, self._cfg.port)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_command(self, device_id: str, command: dict[str, object]) -> None:
        topic = self._cfg.command_topic_tpl.format(device_id=device_id)
        self._client.publish(topic, json.dumps(command, ensure_ascii=False), qos=1)

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            for topic in self._cfg.sensor_topics:
                client.subscribe(topic)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode())
            reading = SensorReading(
                device_id=DeviceId(payload.get("device_id", "unknown")),
                sensor_type=payload.get("type", "unknown"),
                value=float(payload.get("value", 0)),
                unit=payload.get("unit", ""),
                timestamp=time.time(),
            )
            for handler in self._handlers:
                handler(reading)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
