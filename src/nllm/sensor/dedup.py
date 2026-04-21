"""Stream deduplicator — content-fingerprint based duplicate detection."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DedupResult:
    is_duplicate: bool
    fingerprint: str
    first_seen: float = 0.0


class StreamDeduplicator:
    """Time-window content deduplication for IoT data streams."""

    def __init__(self, window_seconds: float = 60.0, max_entries: int = 10_000) -> None:
        self._window = window_seconds
        self._max = max_entries
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check(
        self,
        data: Mapping[str, object],
        key_fields: Sequence[str] | None = None,
    ) -> DedupResult:
        self._evict()
        fp = _fingerprint(data, key_fields)
        now = time.time()

        if fp in self._seen:
            return DedupResult(True, fp, self._seen[fp])

        self._seen[fp] = now
        return DedupResult(False, fp, now)

    @property
    def size(self) -> int:
        return len(self._seen)

    def clear(self) -> None:
        self._seen.clear()

    def _evict(self) -> None:
        cutoff = time.time() - self._window
        while self._seen:
            oldest = next(iter(self._seen))
            if self._seen[oldest] < cutoff:
                self._seen.pop(oldest)
            else:
                break
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)


def _fingerprint(
    data: Mapping[str, object],
    key_fields: Sequence[str] | None,
) -> str:
    subset = {k: data[k] for k in key_fields if k in data} if key_fields else dict(data)
    raw = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
