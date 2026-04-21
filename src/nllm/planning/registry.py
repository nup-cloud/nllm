"""Tool registry — capability discovery for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    domain: str
    fn: Callable[..., Any]
    requires_approval: bool = False
    offline: bool = True


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def by_domain(self, domain: str) -> tuple[ToolDef, ...]:
        return tuple(t for t in self._tools.values() if t.domain == domain)

    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({t.domain for t in self._tools.values()}))

    def invoke(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown_tool:{name}")
        return tool.fn(**kwargs)

    def describe(self, domain: str | None = None) -> str:
        tools = self.by_domain(domain) if domain else tuple(self._tools.values())
        if not tools:
            return "利用可能なツールなし"
        lines = ["利用可能なツール:"]
        for t in sorted(tools, key=lambda x: x.name):
            flags = ""
            if t.requires_approval:
                flags += " [要承認]"
            if t.offline:
                flags += " [オフライン]"
            lines.append(f"  - {t.name}: {t.description}{flags}")
        return "\n".join(lines)
