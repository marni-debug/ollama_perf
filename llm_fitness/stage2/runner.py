from __future__ import annotations

from collections.abc import Callable

from llm_fitness.ollama_client import OllamaClient
from llm_fitness.results import TestResult
from llm_fitness.scoring import TEST_MAX, score_binary, score_test8
from llm_fitness.stage2.agent_loop import MAX_TURNS, run_agent
from llm_fitness.stage2.cases import AgentScenario, scenarios
from llm_fitness.stage2.graders import (
    contradiction_detected,
    discipline_ok,
    invented_session_apis,
    python_sources,
    run_workspace_pytest,
    uses_real_transaction_api,
)
from llm_fitness.stage2.protocol import ToolProtocol
from llm_fitness.stage2.tools import ToolExecutor
from llm_fitness.stage2.workspace import cleanup_workspace, create_workspace
from llm_fitness.stage2.xml_protocol import XmlProtocol

NAMES = {
    "5": "Requirement Compliance",
    "6": "Unnecessary Code Changes",
    "7": "Self-Correction",
    "8": "Tool/Repository Reasoning",
}


def run_stage2(
    client: OllamaClient,
    model: str,
    *,
    protocol: ToolProtocol | None = None,
    max_turns: int = MAX_TURNS,
    keep_workspaces: bool = False,
    on_progress: Callable[[TestResult], None] | None = None,
) -> list[TestResult]:
    proto = protocol or XmlProtocol()
    results: list[TestResult] = []
    for scenario in scenarios():
        batch = _run_scenario(
            client,
            model,
            scenario,
            proto,
            max_turns,
            keep_workspaces,
        )
        for item in batch:
            results.append(item)
            if on_progress:
                on_progress(item)
    return results


def _run_scenario(
    client: OllamaClient,
    model: str,
    scenario: AgentScenario,
    protocol: ToolProtocol,
    max_turns: int,
    keep_workspaces: bool,
) -> list[TestResult]:
    workspace = create_workspace(overlay=scenario.overlay, keep=keep_workspaces)
    before = workspace.snapshot()
    executor = ToolExecutor(workspace.root, workspace.venv_bin)
    agent = run_agent(
        client,
        model,
        executor,
        scenario.task,
        protocol=protocol,
        max_turns=max_turns,
    )
    after = workspace.snapshot()
    pytest_result = run_workspace_pytest(workspace, list(scenario.extra_gold))
    source = python_sources(workspace)
    assistant = "\n".join(
        msg["content"] for msg in agent.transcript if msg.get("role") == "assistant"
    )
    if agent.completion:
        assistant += "\n" + agent.completion

    disc_ok, disc_detail, disc_checks = discipline_ok(before, after, set(scenario.allowed_files))
    invented = invented_session_apis(source)
    real_api = uses_real_transaction_api(source) and not invented
    contradiction = contradiction_detected(assistant)
    ran_tests = executor.ran_pytest()

    common_perf = agent.perf
    extra = {
        "turns": agent.turns,
        "completion": agent.completion,
        "tool_log": agent.tool_log,
        "pytest": pytest_result.output[-4000:],
        "ran_pytest": ran_tests,
        "workspace": str(workspace.root),
    }

    results: list[TestResult] = []
    if "5" in scenario.test_ids:
        passed = pytest_result.passed
        results.append(
            _make(
                "5",
                passed,
                score_binary(passed, TEST_MAX["5"]),
                "health endpoint and existing tests passed" if passed else "tests failed after the change",
                {"pytest": passed, "ran_pytest": ran_tests, **disc_checks},
                assistant,
                common_perf,
                extra,
            )
        )
    if "6" in scenario.test_ids:
        results.append(
            _make(
                "6",
                disc_ok,
                score_binary(disc_ok, TEST_MAX["6"]),
                disc_detail,
                disc_checks,
                assistant,
                common_perf,
                extra,
            )
        )
    if "7" in scenario.test_ids:
        passed = pytest_result.passed and disc_ok
        detail = "fixed failing tests with a targeted change" if passed else (
            "tests still failing" if not pytest_result.passed else disc_detail
        )
        results.append(
            _make(
                "7",
                passed,
                score_binary(passed, TEST_MAX["7"]),
                detail,
                {"pytest": pytest_result.passed, "ran_pytest": ran_tests, **disc_checks},
                assistant,
                common_perf,
                extra,
            )
        )
    if "8" in scenario.test_ids:
        points, critical = score_test8(
            contradiction, real_api, pytest_result.passed, invented
        )
        passed = contradiction and real_api and pytest_result.passed and not critical
        detail_parts = []
        if contradiction:
            detail_parts.append("contradiction noted")
        else:
            detail_parts.append("did not challenge README API")
        if invented:
            detail_parts.append("invented: " + ", ".join(invented))
        elif real_api:
            detail_parts.append("real session transaction API")
        else:
            detail_parts.append("no real transaction API")
        detail_parts.append("tests pass" if pytest_result.passed else "tests fail")
        results.append(
            TestResult(
                id="8",
                name=NAMES["8"],
                stage=2,
                passed=passed,
                points=points,
                max_points=TEST_MAX["8"],
                detail="; ".join(detail_parts),
                checks={
                    "contradiction": contradiction,
                    "real_api": real_api,
                    "tests": pytest_result.passed,
                    "invented_apis": invented,
                    "ran_pytest": ran_tests,
                    **disc_checks,
                },
                critical_hallucination=critical,
                invented_apis=invented,
                response=assistant,
                perf=common_perf,
                scored=True,
            )
        )
        results[-1].checks["workspace"] = extra["workspace"]
        results[-1].checks["turns"] = extra["turns"]

    if not keep_workspaces:
        cleanup_workspace(workspace)
    return results


def _make(
    test_id: str,
    passed: bool,
    points: float,
    detail: str,
    checks: dict,
    response: str,
    perf,
    extra: dict,
) -> TestResult:
    merged = dict(checks)
    merged["turns"] = extra.get("turns")
    merged["ran_pytest"] = extra.get("ran_pytest")
    return TestResult(
        id=test_id,
        name=NAMES[test_id],
        stage=2,
        passed=passed,
        points=points,
        max_points=TEST_MAX[test_id],
        detail=detail,
        checks=merged,
        response=response,
        perf=perf,
        scored=True,
    )
