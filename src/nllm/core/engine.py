"""Inference engine protocol — pluggable backend abstraction.

Concrete implementations live in nllm.adapters (ollama, llama_cpp).
Domain code depends only on this Protocol, never on a concrete backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationParams:
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    tokens_used: int = 0


@runtime_checkable
class InferenceEngine(Protocol):
    """Anything that can produce text from a prompt."""

    def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult: ...

    def is_available(self) -> bool: ...
