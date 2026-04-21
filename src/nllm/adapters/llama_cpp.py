"""llama-cpp-python adapter — direct GGUF inference for embedded deployments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nllm.core.engine import GenerationParams, GenerationResult, InferenceEngine


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    model_path: Path
    n_ctx: int = 4096
    n_threads: int | None = None
    n_gpu_layers: int = 0


class LlamaCppEngine:
    """InferenceEngine backed by llama-cpp-python."""

    def __init__(self, config: LlamaCppConfig) -> None:
        self._cfg = config
        self._llm: Any = None
        self._error: str | None = None

    def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        llm = self._ensure()
        if llm is None:
            return GenerationResult("")

        p = params or GenerationParams()
        out = llm(
            prompt,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            stop=list(p.stop) if p.stop else None,
        )
        text = out["choices"][0]["text"].strip()
        tokens = out.get("usage", {}).get("total_tokens", 0)
        return GenerationResult(text, tokens)

    def is_available(self) -> bool:
        return self._ensure() is not None

    def _ensure(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self._error is not None:
            return None

        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError:
            self._error = "llama-cpp-python not installed"
            return None

        if not self._cfg.model_path.exists():
            self._error = f"model not found: {self._cfg.model_path}"
            return None

        kwargs: dict[str, Any] = {
            "model_path": str(self._cfg.model_path),
            "n_ctx": self._cfg.n_ctx,
            "n_gpu_layers": self._cfg.n_gpu_layers,
            "verbose": False,
        }
        if self._cfg.n_threads is not None:
            kwargs["n_threads"] = self._cfg.n_threads

        self._llm = Llama(**kwargs)
        return self._llm
