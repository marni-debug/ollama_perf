from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_fitness.engine import expected_scored, run_fitness
from llm_fitness.ollama_client import OllamaClient, OllamaError
from llm_fitness.persist import load_run, save_run
from llm_fitness.report import progress_line, render_compare, render_footer, render_header
from llm_fitness.results import TestResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Coding-Agent Fitness Test for local Ollama models",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="Ollama model tag (repeatable)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "knowledge", "agent"),
        default="full",
        help="full = 8 tests (default), knowledge = stage 1, agent = stage 2",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="JSON",
        help="Compare previously saved result JSON files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Also print the result JSON to stdout",
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compare:
        runs = [load_run(Path(path)) for path in args.compare]
        sys.stdout.write(render_compare(runs))
        return 0
    if not args.models:
        parser.error("the following arguments are required: -m/--model")

    client = OllamaClient(base_url=args.host)
    exit_code = 0
    for model in args.models:
        try:
            result = _run_one(client, model, args)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        path = save_run(result, directory=args.results_dir)
        print(f"Wrote {path}", file=sys.stderr)
        if args.as_json:
            sys.stdout.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return exit_code


def _run_one(client: OllamaClient, model: str, args: argparse.Namespace):
    total = expected_scored(args.mode)
    context = "unknown"
    try:
        from llm_fitness.ollama_client import NUM_CTX, format_num_ctx, model_max_context

        effective = format_num_ctx(NUM_CTX)
        model_max = model_max_context(client.show(model))
        if model_max is not None:
            context = f"{effective} (model max {format_num_ctx(model_max)})"
        else:
            context = effective
    except OllamaError:
        raise
    print(render_header(model, context), flush=True)
    index = 0

    def on_progress(test: TestResult) -> None:
        nonlocal index
        if test.scored:
            index += 1
            print(progress_line(index, total, test.name, test.passed), flush=True)
        else:
            from llm_fitness.report import _dots

            status = "PASS" if test.passed else "FAIL"
            print(f"[extra] {_dots(test.name, status)}", flush=True)

    result = run_fitness(
        client,
        model,
        mode=args.mode,
        max_turns=args.max_turns,
        on_progress=on_progress,
    )
    print(render_footer(result), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
