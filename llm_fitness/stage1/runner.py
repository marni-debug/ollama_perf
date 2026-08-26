from __future__ import annotations

from collections.abc import Callable

from llm_fitness.ollama_client import OllamaClient
from llm_fitness.results import TestResult
from llm_fitness.scoring import TEST_MAX, score_binary
from llm_fitness.stage1.cases import KnowledgeCase, knowledge_cases

SYSTEM_PROMPT = (
    "You are a helpful software engineer answering a colleague. "
    "Be precise about Python runtime behavior and standard-library APIs. "
    "If a claim or API looks wrong, say so."
)


def run_stage1(
    client: OllamaClient,
    model: str,
    on_progress: Callable[[TestResult], None] | None = None,
) -> list[TestResult]:
    results: list[TestResult] = []
    for case in knowledge_cases():
        result = _run_case(client, model, case)
        results.append(result)
        if on_progress:
            on_progress(result)
    return results


def _run_case(client: OllamaClient, model: str, case: KnowledgeCase) -> TestResult:
    response = client.chat(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.prompt},
        ],
    )
    grade = case.grader(response.content)
    max_points = TEST_MAX.get(case.id, 0.0) if case.scored else 10.0
    points = score_binary(grade.passed, max_points) if case.scored else (10.0 if grade.passed else 0.0)
    return TestResult(
        id=case.id,
        name=case.name,
        stage=1,
        passed=grade.passed,
        points=points,
        max_points=max_points,
        detail=grade.detail,
        checks=grade.checks,
        response=response.content,
        perf=[response.perf],
        scored=case.scored,
    )
