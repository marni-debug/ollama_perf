from __future__ import annotations

from llm_fitness.ollama_client import aggregate_perf
from llm_fitness.results import PerfSample, RunResult, TestResult

TEST_MAX = {
    "1": 12.0,
    "2": 12.0,
    "3": 12.0,
    "4": 12.0,
    "5": 12.0,
    "6": 12.0,
    "7": 12.0,
    "8": 16.0,
}

CRITICAL_PENALTY = 15.0

VERDICTS = (
    (90, "EXCELLENT CODING AGENT"),
    (75, "GOOD CODING AGENT"),
    (60, "USABLE WITH SUPERVISION"),
    (0, "WEAK / NOT SUITABLE AS AUTONOMOUS AGENT"),
)


def score_binary(passed: bool, max_points: float) -> float:
    return max_points if passed else 0.0


def score_test8(
    contradiction: bool,
    real_api: bool,
    tests_pass: bool,
    invented_apis: list[str],
) -> tuple[float, bool]:
    points = 0.0
    if contradiction:
        points += 6.0
    if real_api:
        points += 6.0
    if tests_pass:
        points += 4.0
    critical = bool(invented_apis)
    return points, critical


def _ratio(tests: list[TestResult], ids: tuple[str, ...]) -> float | None:
    selected = [t for t in tests if t.id in ids and t.passed is not None]
    if not selected:
        return None
    scores = []
    for test in selected:
        if test.max_points <= 0:
            continue
        scores.append(10.0 * test.points / test.max_points)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _file_discipline(test: TestResult) -> float | None:
    """0–10 from Diff-Disziplin (illegal files), not from trap/API points."""
    if test.passed is None:
        return None
    checks = test.checks or {}
    if "illegal" in checks:
        return 0.0 if checks["illegal"] else 10.0
    if test.max_points <= 0:
        return None
    return 10.0 * test.points / test.max_points


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def dimension_scores(tests: list[TestResult]) -> dict[str, float | None]:
    by_id = {t.id: t for t in tests}
    hall_ids = ["1", "8"]
    if "false_premise" in by_id and by_id["false_premise"].passed is not None:
        hall_ids.append("false_premise")
    discipline = []
    for test_id in ("6", "8"):
        test = by_id.get(test_id)
        if test is None:
            continue
        score = _file_discipline(test)
        if score is not None:
            discipline.append(score)
    return {
        "hallucination_resistance": _ratio(tests, tuple(hall_ids)),
        "code_reasoning": _ratio(tests, ("2", "3", "4")),
        "instruction_following": _ratio(tests, ("5",)),
        "agent_discipline": _mean(discipline),
        "self_correction": _ratio(tests, ("7",)),
    }


def verdict_for(total: float, mode: str) -> str | None:
    if mode != "full":
        return None
    for threshold, label in VERDICTS:
        if total >= threshold:
            return label
    return VERDICTS[-1][1]


def build_run_result(
    *,
    model: str,
    ollama_version: str,
    context: str,
    mode: str,
    tests: list[TestResult],
    extra: dict | None = None,
) -> RunResult:
    scored = [t for t in tests if t.scored and t.passed is not None]
    total = sum(t.points for t in scored)
    total_max = sum(t.max_points for t in scored)
    invented: list[str] = []
    critical = False
    for test in tests:
        if test.critical_hallucination:
            critical = True
            invented.extend(test.invented_apis)
    penalty = CRITICAL_PENALTY if critical else 0.0
    if critical:
        total = max(0.0, total - penalty)

    samples: list[PerfSample] = []
    for test in tests:
        samples.extend(test.perf)
    avg_s, tps = aggregate_perf(samples)

    return RunResult(
        model=model,
        ollama_version=ollama_version,
        context=context,
        mode=mode,
        tests=tests,
        total=total,
        total_max=total_max,
        penalty=penalty,
        dimensions=dimension_scores(tests),
        verdict=verdict_for(total, mode),
        avg_response_s=avg_s,
        tokens_per_sec=tps,
        critical_hallucination=critical,
        invented_apis=invented,
        extra=extra or {},
    )
