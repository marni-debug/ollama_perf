"""Minimal YAML subset: mappings, nested mappings, string lists, scalars."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class YamlError(ValueError):
    pass


def load_yaml_file(path: Path) -> Any:
    return load_yaml(path.read_text(encoding="utf-8"))


def load_yaml(text: str) -> Any:
    lines = _logical_lines(text)
    if not lines:
        return {}
    value, index = _parse_value(lines, 0, lines[0][0])
    if index != len(lines):
        raise YamlError(f"unexpected content at {lines[index][1]!r}")
    return value


def _logical_lines(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if "\t" in stripped[: len(stripped) - len(stripped.lstrip())]:
            raise YamlError("tabs are not supported")
        result.append((indent, stripped.strip()))
    return result


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and in_double:
            i += 2
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
        i += 1
    return line


def _parse_value(lines: list[tuple[int, str]], index: int, min_indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index
    indent, content = lines[index]
    if indent < min_indent:
        return None, index
    if content.startswith("- "):
        return _parse_list(lines, index, indent)
    if ":" in content:
        return _parse_map(lines, index, indent)
    return _parse_scalar(content), index + 1


def _parse_map(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"unexpected indent at {content!r}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise YamlError(f"expected key: value, got {content!r}")
        key, rest = content.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest:
            result[key] = _parse_scalar(rest)
            continue
        if index < len(lines) and lines[index][0] > indent:
            child_indent = lines[index][0]
            value, index = _parse_value(lines, index, child_indent)
            result[key] = value
        else:
            result[key] = None
    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"unexpected indent at {content!r}")
        if not content.startswith("- "):
            break
        item = content[2:].strip()
        index += 1
        if item:
            result.append(_parse_scalar(item))
            continue
        if index < len(lines) and lines[index][0] > indent:
            child_indent = lines[index][0]
            value, index = _parse_value(lines, index, child_indent)
            result.append(value)
        else:
            result.append(None)
    return result, index


def _parse_scalar(text: str) -> Any:
    if text in ("", "null", "~"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        inner = text[1:-1]
        if text[0] == '"':
            inner = (
                inner.replace("\\\\", "\0")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\0", "\\")
            )
        return inner
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text
