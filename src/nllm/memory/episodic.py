"""Episodic memory — time-ordered event sequences for context recall.

Stores complete interaction episodes (session-level) that can be
recalled to provide context for similar future situations.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    title: str
    events: tuple[EpisodeEvent, ...]
    outcome: str  # "success" | "failure" | "partial"
    domain: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    timestamp: float
    event_type: str  # "input" | "command" | "response" | "error" | "approval"
    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class EpisodicMemory:
    """Stores complete interaction episodes for contextual recall.

    Use cases:
    - "Last time we flew the drone in rain, what happened?"
    - "What commands worked for the warehouse inspection?"
    - Provide context for similar future tasks
    """

    def __init__(self, max_episodes: int = 50, persist_path: Path | None = None) -> None:
        self._episodes: deque[Episode] = deque(maxlen=max_episodes)
        self._current_events: list[EpisodeEvent] = []
        self._current_title: str = ""
        self._current_domain: str = ""
        self._episode_counter = 0
        self._path = persist_path
        if self._path and self._path.exists():
            self._load()

    # ── Recording ────────────────────────────────────────────────────

    def begin_episode(self, title: str, domain: str = "") -> None:
        self._current_title = title
        self._current_domain = domain
        self._current_events = []

    def record_event(
        self,
        event_type: str,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._current_events.append(EpisodeEvent(
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            metadata=metadata or {},
        ))

    def end_episode(self, outcome: str, tags: Sequence[str] = ()) -> Episode:
        self._episode_counter += 1
        episode = Episode(
            episode_id=f"ep_{self._episode_counter:05d}",
            title=self._current_title,
            events=tuple(self._current_events),
            outcome=outcome,
            domain=self._current_domain,
            started_at=self._current_events[0].timestamp if self._current_events else time.time(),
            ended_at=time.time(),
            tags=tuple(tags),
        )
        self._episodes.append(episode)
        self._current_events = []
        self._current_title = ""
        return episode

    # ── Recall ───────────────────────────────────────────────────────

    def recall_by_domain(self, domain: str) -> tuple[Episode, ...]:
        return tuple(ep for ep in self._episodes if ep.domain == domain)

    def recall_by_tag(self, tag: str) -> tuple[Episode, ...]:
        return tuple(ep for ep in self._episodes if tag in ep.tags)

    def recall_by_outcome(self, outcome: str) -> tuple[Episode, ...]:
        return tuple(ep for ep in self._episodes if ep.outcome == outcome)

    def recall_recent(self, n: int = 5) -> tuple[Episode, ...]:
        return tuple(list(self._episodes)[-n:])

    def recall_similar(self, query: str, top_k: int = 3) -> tuple[Episode, ...]:
        """Find episodes with titles or events matching the query."""
        query_lower = query.lower()
        scored: list[tuple[float, Episode]] = []

        for ep in self._episodes:
            score = 0.0
            if query_lower in ep.title.lower():
                score += 2.0
            for event in ep.events:
                if query_lower in event.content.lower():
                    score += 0.5
            if any(query_lower in tag.lower() for tag in ep.tags):
                score += 1.0
            if score > 0:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(ep for _, ep in scored[:top_k])

    # ── Stats ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def is_recording(self) -> bool:
        return len(self._current_events) > 0

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "events": [
                    {"timestamp": e.timestamp, "event_type": e.event_type,
                     "content": e.content, "metadata": dict(e.metadata)}
                    for e in ep.events
                ],
                "outcome": ep.outcome,
                "domain": ep.domain,
                "started_at": ep.started_at,
                "ended_at": ep.ended_at,
                "tags": list(ep.tags),
            }
            for ep in self._episodes
        ]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            for item in data:
                events = tuple(
                    EpisodeEvent(e["timestamp"], e["event_type"], e["content"], e.get("metadata", {}))
                    for e in item["events"]
                )
                ep = Episode(
                    episode_id=item["episode_id"],
                    title=item["title"],
                    events=events,
                    outcome=item["outcome"],
                    domain=item.get("domain", ""),
                    started_at=item.get("started_at", 0),
                    ended_at=item.get("ended_at", 0),
                    tags=tuple(item.get("tags", ())),
                )
                self._episodes.append(ep)
                self._episode_counter = max(self._episode_counter, int(ep.episode_id.split("_")[1]))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def clear(self) -> None:
        self._episodes.clear()
        self._current_events = []
