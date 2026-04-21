"""Agent memory — conversation and device state repository."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    entry_type: str  # "conversation" | "command" | "alert"
    content: Mapping[str, object]
    timestamp: float = field(default_factory=time.time)
    tags: tuple[str, ...] = ()


class AgentMemory:
    """Rolling-window memory with optional persistence."""

    def __init__(self, max_history: int = 100, persist_path: Path | None = None) -> None:
        self._entries: deque[MemoryEntry] = deque(maxlen=max_history)
        self._devices: dict[str, dict[str, object]] = {}
        self._path = persist_path
        if self._path and self._path.exists():
            self._load()

    # ── Write ────────────────────────────────────────────────────────

    def add_conversation(self, role: str, text: str) -> None:
        self._entries.append(MemoryEntry("conversation", {"role": role, "text": text}))

    def add_command(self, action: str, params: Mapping[str, object], success: bool) -> None:
        self._entries.append(MemoryEntry("command", {"action": action, "params": dict(params), "success": success}, tags=("command",)))

    def add_alert(self, alert_type: str, details: Mapping[str, object]) -> None:
        self._entries.append(MemoryEntry("alert", {"alert_type": alert_type, **details}, tags=("alert",)))

    def update_device(self, device_id: str, state: Mapping[str, object]) -> None:
        self._devices[device_id] = {**state, "last_updated": time.time()}

    # ── Read ─────────────────────────────────────────────────────────

    def device_state(self, device_id: str) -> Mapping[str, object] | None:
        return self._devices.get(device_id)

    def recent_conversations(self, n: int = 10) -> tuple[Mapping[str, object], ...]:
        return tuple(e.content for e in self._entries if e.entry_type == "conversation")[-n:]

    def recent_commands(self, n: int = 10) -> tuple[Mapping[str, object], ...]:
        return tuple(e.content for e in self._entries if e.entry_type == "command")[-n:]

    def context_summary(self) -> str:
        parts = ["=== 最近の活動 ==="]
        for entry in list(self._entries)[-10:]:
            match entry.entry_type:
                case "conversation":
                    parts.append(f"[{entry.content.get('role', '?')}] {str(entry.content.get('text', ''))[:80]}")
                case "command":
                    ok = "OK" if entry.content.get("success") else "FAIL"
                    parts.append(f"[CMD] {entry.content.get('action', '?')} -> {ok}")
                case "alert":
                    parts.append(f"[ALERT] {entry.content.get('alert_type', '?')}")
        return "\n".join(parts)

    # ── Lifecycle ────────────────────────────────────────────────────

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [{"type": e.entry_type, "content": dict(e.content), "timestamp": e.timestamp, "tags": e.tags} for e in self._entries],
            "devices": self._devices,
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self._entries.clear()
        self._devices.clear()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            for e in data.get("entries", []):
                self._entries.append(MemoryEntry(e["type"], e["content"], e.get("timestamp", 0), tuple(e.get("tags", ()))))
            self._devices = data.get("devices", {})
        except (json.JSONDecodeError, KeyError):
            pass
