from __future__ import annotations

from collections.abc import Callable

from llm_fitness.ollama_client import (
    NUM_CTX,
    OllamaClient,
    format_num_ctx,
    model_max_context,
)
from llm_fitness.results import RunResult, TestResult
from llm_fitness.scoring import build_run_result
from llm_fitness.stage1.runner import run_stage1
from llm_fitness.stage2.agent_loop import MAX_TURNS
from llm_fitness.stage2.runner import run_stage2


def expected_scored(mode: str) -> int:
    if mode == "knowledge":
        return 4
    if mode == "agent":
        return 4
    return 8


def run_fitness(
    client: OllamaClient,
    model: str,
    mode: str = "full",
    max_turns: int = MAX_TURNS,
    on_progress: Callable[[TestResult], None] | None = None,
) -> RunResult:
    show = client.show(model)
    tests: list[TestResult] = []
    if mode in {"full", "knowledge"}:
        tests.extend(run_stage1(client, model, on_progress=on_progress))
    if mode in {"full", "agent"}:
        tests.extend(
            run_stage2(
                client,
                model,
                max_turns=max_turns,
                on_progress=on_progress,
            )
        )
    result = build_run_result(
        model=model,
        ollama_version=client.version(),
        context=format_num_ctx(NUM_CTX),
        mode=mode,
        tests=tests,
    )
    result.context_effective = NUM_CTX
    result.context_model_max = model_max_context(show)
    return result
