"""Eval runner — loads test data, runs inference, computes metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from nllm.eval.metrics import (
    command_match_rate,
    exact_match_rate,
    json_valid_rate,
    safety_rejection_rate,
)


@runtime_checkable
class InferenceEngine(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class EvalSample:
    prompt: str
    reference: str


@dataclass(frozen=True, slots=True)
class EvalReport:
    exact_match: float
    command_match: float
    safety_rejection: float
    json_valid: float
    total_samples: int
    latency_ms_avg: float


class EvalRunner:
    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def load_samples(self, path: Path) -> tuple[EvalSample, ...]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(EvalSample(prompt=obj["prompt"], reference=obj["reference"]))
        return tuple(samples)

    def run(self, samples: Sequence[EvalSample]) -> EvalReport:
        predictions: list[str] = []
        references: list[str] = []
        total_ms: float = 0.0

        for sample in samples:
            t0 = time.monotonic()
            pred = self._engine.generate(sample.prompt)
            elapsed = (time.monotonic() - t0) * 1000
            total_ms += elapsed
            predictions.append(pred)
            references.append(sample.reference)

        n = len(samples)
        return EvalReport(
            exact_match=exact_match_rate(predictions, references),
            command_match=command_match_rate(predictions, references),
            safety_rejection=safety_rejection_rate(predictions),
            json_valid=json_valid_rate(predictions),
            total_samples=n,
            latency_ms_avg=total_ms / n if n else 0.0,
        )
