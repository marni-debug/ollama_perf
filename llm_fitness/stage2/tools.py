from __future__ import annotations

from pathlib import Path

from llm_fitness.stage2.protocol import ToolCall, ToolResult
from llm_fitness.stage2.sandbox import (
    SandboxError,
    resolve_in_workspace,
    run_command,
)

SKIP_NAMES = {".venv", "__pycache__", ".git", ".pytest_cache"}


class ToolExecutor:
    def __init__(self, workspace: Path, venv_bin: Path, timeout: int = 30):
        self.workspace = workspace.resolve()
        self.venv_bin = venv_bin.resolve()
        self.timeout = timeout
        self.log: list[dict[str, str]] = []

    def execute(self, call: ToolCall) -> ToolResult:
        handler = {
            "read_file": self._read_file,
            "write_to_file": self._write_to_file,
            "replace_in_file": self._replace_in_file,
            "list_files": self._list_files,
            "execute_command": self._execute_command,
            "attempt_completion": self._complete,
        }.get(call.name)
        if handler is None:
            result = ToolResult(call.name, False, f"unknown tool: {call.name}")
        else:
            try:
                result = handler(call.arguments)
            except SandboxError as exc:
                result = ToolResult(call.name, False, str(exc))
            except OSError as exc:
                result = ToolResult(call.name, False, str(exc))
        self.log.append(
            {
                "tool": call.name,
                "ok": str(result.ok),
                "output": result.output[:2000],
                "args": repr({k: v[:200] for k, v in call.arguments.items()}),
            }
        )
        return result

    def _read_file(self, args: dict[str, str]) -> ToolResult:
        path = resolve_in_workspace(self.workspace, args.get("path", ""))
        if not path.is_file():
            return ToolResult("read_file", False, f"not a file: {args.get('path')}")
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(self.workspace)
        numbered = "\n".join(f"{i:4d}|{line}" for i, line in enumerate(text.splitlines(), 1))
        return ToolResult("read_file", True, f"{rel}\n{numbered}")

    def _write_to_file(self, args: dict[str, str]) -> ToolResult:
        path = resolve_in_workspace(self.workspace, args.get("path", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.get("content", ""), encoding="utf-8")
        return ToolResult("write_to_file", True, f"wrote {path.relative_to(self.workspace)}")

    def _replace_in_file(self, args: dict[str, str]) -> ToolResult:
        path = resolve_in_workspace(self.workspace, args.get("path", ""))
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        if not old:
            return ToolResult("replace_in_file", False, "old_string is empty")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return ToolResult("replace_in_file", False, "old_string not found")
        if count > 1:
            return ToolResult(
                "replace_in_file",
                False,
                f"old_string found {count} times; make it unique",
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult("replace_in_file", True, f"updated {path.relative_to(self.workspace)}")

    def _list_files(self, args: dict[str, str]) -> ToolResult:
        target = resolve_in_workspace(self.workspace, args.get("path") or ".")
        if not target.exists():
            return ToolResult("list_files", False, "path not found")
        if target.is_file():
            return ToolResult("list_files", True, str(target.relative_to(self.workspace)))
        names: list[str] = []
        for item in sorted(target.rglob("*")):
            if any(part in SKIP_NAMES for part in item.relative_to(self.workspace).parts):
                continue
            rel = item.relative_to(self.workspace)
            suffix = "/" if item.is_dir() else ""
            names.append(f"{rel}{suffix}")
        return ToolResult("list_files", True, "\n".join(names) or ".")

    def _execute_command(self, args: dict[str, str]) -> ToolResult:
        command = args.get("command", "").strip()
        result = run_command(self.workspace, self.venv_bin, command, timeout=self.timeout)
        output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
        if result.rejected:
            return ToolResult("execute_command", False, output or result.stderr)
        ok = result.returncode == 0
        header = f"exit {result.returncode}"
        return ToolResult("execute_command", ok, f"{header}\n{output}".strip())

    def _complete(self, args: dict[str, str]) -> ToolResult:
        return ToolResult("attempt_completion", True, args.get("result", ""))

    def ran_pytest(self) -> bool:
        for entry in self.log:
            if entry["tool"] != "execute_command":
                continue
            blob = entry.get("args", "").lower() + entry.get("output", "").lower()
            if "pytest" in blob:
                return True
        return False
