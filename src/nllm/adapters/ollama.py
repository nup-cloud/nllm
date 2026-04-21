"""Ollama adapter — local model serving via Ollama CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from nllm.core.engine import GenerationParams, GenerationResult, InferenceEngine


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    model: str = "nllm"
    timeout_sec: int = 60


class OllamaEngine:
    """InferenceEngine backed by Ollama."""

    def __init__(self, config: OllamaConfig = OllamaConfig()) -> None:
        self._cfg = config

    def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        try:
            result = subprocess.run(
                ["ollama", "run", self._cfg.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout_sec,
            )
            if result.returncode != 0:
                return GenerationResult("")
            return GenerationResult(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return GenerationResult("")

    def is_available(self) -> bool:
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
