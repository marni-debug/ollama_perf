from __future__ import annotations

import re

from llm_fitness.stage2.protocol import ToolCall, ToolResult

TOOLS = (
    "read_file",
    "write_to_file",
    "replace_in_file",
    "list_files",
    "execute_command",
    "attempt_completion",
)

_BLOCK = re.compile(
    r"<(?P<name>" + "|".join(TOOLS) + r")>(?P<body>.*?)</(?P=name)>",
    re.DOTALL | re.IGNORECASE,
)
_TAG = re.compile(r"<(?P<name>[a-zA-Z_]+)>(?P<body>.*?)</(?P=name)>", re.DOTALL)


SYSTEM_PROMPT = """You are a software engineer working in a local Python project.
The project root is the current working directory.

Solve the user's request using tools. Put tool calls in this XML form (you may use several):

<read_file>
<path>relative/path</path>
</read_file>

<write_to_file>
<path>relative/path</path>
<content>
file contents
</content>
</write_to_file>

<replace_in_file>
<path>relative/path</path>
<old_string>
exact text to find
</old_string>
<new_string>
replacement
</new_string>
</replace_in_file>

<list_files>
<path>.</path>
</list_files>

<execute_command>
<command>pytest</command>
</execute_command>

When the task is done:

<attempt_completion>
<result>
short summary of what you did
</result>
</attempt_completion>

Guidelines:
- Prefer small, targeted edits. Do not rewrite unrelated files.
- Dependencies are already installed. Do not install packages.
- Run tests with pytest after behavior changes.
- Stay inside the project directory.
"""


class XmlProtocol:
    def parse(self, assistant_text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for match in _BLOCK.finditer(assistant_text or ""):
            name = match.group("name").lower()
            body = match.group("body").strip()
            args = _parse_args(name, body)
            calls.append(ToolCall(name=name, arguments=args))
        return calls

    def format_result(self, result: ToolResult) -> str:
        status = "OK" if result.ok else "ERROR"
        return f"<tool_result name=\"{result.name}\" status=\"{status}\">\n{result.output}\n</tool_result>"

    def system_prompt(self, workspace_hint: str) -> str:
        return SYSTEM_PROMPT + f"\nWorking directory: {workspace_hint}\n"


def _parse_args(name: str, body: str) -> dict[str, str]:
    tagged: dict[str, str] = {}
    for match in _TAG.finditer(body):
        tagged[match.group("name").lower()] = match.group("body").strip("\n")
    if name == "read_file":
        return {"path": tagged.get("path", body)}
    if name == "write_to_file":
        return {"path": tagged.get("path", ""), "content": tagged.get("content", "")}
    if name == "replace_in_file":
        return {
            "path": tagged.get("path", ""),
            "old_string": tagged.get("old_string", ""),
            "new_string": tagged.get("new_string", ""),
        }
    if name == "list_files":
        return {"path": tagged.get("path", body or ".")}
    if name == "execute_command":
        return {"command": tagged.get("command", body)}
    if name == "attempt_completion":
        return {"result": tagged.get("result", body)}
    return tagged
