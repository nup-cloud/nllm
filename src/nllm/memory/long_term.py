"""Long-term memory — persistent knowledge store with semantic retrieval.

Stores learned facts, device patterns, user preferences, and operational
knowledge that persists across sessions and informs future decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Mapping, Sequence

from nllm.types import DeviceId, Ok, Err, Result


@unique
class MemoryType(Enum):
    FACT = "fact"              # Learned fact ("this device runs hot above 30C")
    PREFERENCE = "preference"  # User/operator preference ("always confirm before takeoff")
    PATTERN = "pattern"        # Detected operational pattern ("sensor spikes at 3pm daily")
    INCIDENT = "incident"      # Past incident record ("motor failure on 2026-03-15")
    SKILL = "skill"            # Learned skill reference ("drone_inspection requires RTH after 20min")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: str
    memory_type: MemoryType
    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    record: MemoryRecord
    score: float


class LongTermMemory:
    """Persistent memory store with keyword-based retrieval.

    Stores knowledge that outlives a single session:
    - Device behavior patterns
    - Operator preferences
    - Past incidents and resolutions
    - Learned operational rules
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._path = persist_path
        if self._path and self._path.exists():
            self._load()

    # ── Store ────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        record_id = _generate_id(content)

        # Update if exists (increment access)
        if record_id in self._records:
            existing = self._records[record_id]
            updated = MemoryRecord(
                record_id=existing.record_id,
                memory_type=existing.memory_type,
                content=existing.content,
                metadata=existing.metadata,
                tags=existing.tags,
                created_at=existing.created_at,
                last_accessed=time.time(),
                access_count=existing.access_count + 1,
                relevance_score=existing.relevance_score,
            )
            self._records[record_id] = updated
            return updated

        record = MemoryRecord(
            record_id=record_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            tags=tuple(tags),
        )
        self._records[record_id] = record
        return record

    # ── Retrieve ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        memory_type: MemoryType | None = None,
        tags: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> tuple[SearchResult, ...]:
        """Keyword-based search with recency and access frequency scoring."""
        candidates = list(self._records.values())

        if memory_type:
            candidates = [r for r in candidates if r.memory_type == memory_type]

        if tags:
            tag_set = set(tags)
            candidates = [r for r in candidates if tag_set & set(r.tags)]

        scored: list[SearchResult] = []
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for record in candidates:
            score = _compute_relevance(record, query_tokens, query_lower)
            if score > 0:
                scored.append(SearchResult(record=record, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return tuple(scored[:top_k])

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def get_by_type(self, memory_type: MemoryType) -> tuple[MemoryRecord, ...]:
        return tuple(r for r in self._records.values() if r.memory_type == memory_type)

    def get_by_tag(self, tag: str) -> tuple[MemoryRecord, ...]:
        return tuple(r for r in self._records.values() if tag in r.tags)

    # ── Delete ───────────────────────────────────────────────────────

    def forget(self, record_id: str) -> bool:
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False

    def forget_by_tag(self, tag: str) -> int:
        to_remove = [rid for rid, r in self._records.items() if tag in r.tags]
        for rid in to_remove:
            del self._records[rid]
        return len(to_remove)

    # ── Stats ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._records)

    def summary(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for r in self._records.values():
            counts[r.memory_type.value] = counts.get(r.memory_type.value, 0) + 1
        return counts

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "record_id": r.record_id,
                "memory_type": r.memory_type.value,
                "content": r.content,
                "metadata": dict(r.metadata),
                "tags": list(r.tags),
                "created_at": r.created_at,
                "last_accessed": r.last_accessed,
                "access_count": r.access_count,
                "relevance_score": r.relevance_score,
            }
            for r in self._records.values()
        ]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            for item in data:
                record = MemoryRecord(
                    record_id=item["record_id"],
                    memory_type=MemoryType(item["memory_type"]),
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    tags=tuple(item.get("tags", ())),
                    created_at=item.get("created_at", 0),
                    last_accessed=item.get("last_accessed", 0),
                    access_count=item.get("access_count", 0),
                    relevance_score=item.get("relevance_score", 1.0),
                )
                self._records[record.record_id] = record
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def clear(self) -> None:
        self._records.clear()


# ── Pure helpers ─────────────────────────────────────────────────────

def _generate_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _compute_relevance(
    record: MemoryRecord,
    query_tokens: set[str],
    query_lower: str,
) -> float:
    """Score relevance based on keyword match, recency, and access frequency."""
    content_lower = record.content.lower()
    tag_text = " ".join(record.tags).lower()

    # Keyword match
    content_tokens = set(content_lower.split())
    overlap = query_tokens & (content_tokens | set(tag_text.split()))
    if not overlap and query_lower not in content_lower:
        return 0.0

    keyword_score = len(overlap) / max(len(query_tokens), 1)

    # Substring match bonus
    if query_lower in content_lower:
        keyword_score += 0.5

    # Recency decay (half-life: 7 days)
    age_days = (time.time() - record.last_accessed) / 86400
    recency_score = 0.5 ** (age_days / 7)

    # Access frequency bonus (capped)
    frequency_score = min(record.access_count / 10, 1.0) * 0.3

    return keyword_score + recency_score * 0.3 + frequency_score
