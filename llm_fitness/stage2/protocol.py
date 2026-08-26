from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str


class ToolProtocol(Protocol):
    def parse(self, assistant_text: str) -> list[ToolCall]:
        """Extract tool calls from a model response."""

    def format_result(self, result: ToolResult) -> str:
        """Render a tool result back into the conversation."""

    def system_prompt(self, workspace_hint: str) -> str:
        """Blind coding-assistant system prompt for this protocol."""
