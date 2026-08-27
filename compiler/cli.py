from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from compiler.analyzer import CompilerError
from compiler.compiler import compile_prompt, explain_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a deterministic coding-agent prompt from fitness JSON")
    parser.add_argument("--fitness", type=Path, required=True, help="Fitness result JSON")
    parser.add_argument("--task", type=Path, required=True, help="Task markdown/text file")
    parser.add_argument("--output", type=Path, default=None, help="Write prompt to this path")
    parser.add_argument("--format", default="markdown", help="Output format (only markdown is supported)")
    parser.add_argument("--profile", action="store_true", help="Also print the capability profile JSON to stdout")
    parser.add_argument("--explain", action="store_true", help="Also print per-category rule selection to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.format != "markdown":
        print("error: only markdown format is supported", file=sys.stderr)
        return 1
    try:
        result = compile_prompt(args.fitness, args.task)
    except CompilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output is not None:
        try:
            _atomic_write(args.output, result.prompt)
        except Exception as exc:
            print(f"error: cannot write output file: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(result.prompt)
    if args.profile:
        sys.stdout.write(json.dumps(result.profile.to_dict(), indent=2, sort_keys=True) + "\n")
    if args.explain:
        sys.stdout.write(explain_text(result))
    return 0


def _atomic_write(path: Path, text: str) -> None:
    if path.name == "":
        raise ValueError(f"output path has no filename: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
