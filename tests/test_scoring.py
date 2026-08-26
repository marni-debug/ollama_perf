from llm_fitness.results import TestResult
from llm_fitness.scoring import build_run_result, score_test8, verdict_for


def _t(id: str, passed: bool, points: float, max_points: float, scored: bool = True, **kwargs) -> TestResult:
    return TestResult(
        id=id,
        name=id,
        stage=1,
        passed=passed,
        points=points,
        max_points=max_points,
        scored=scored,
        **kwargs,
    )


def test_test8_three_aspects():
    points, critical = score_test8(True, True, True, [])
    assert points == 16
    assert not critical
    points, critical = score_test8(False, True, True, ["session.transact()"])
    assert points == 10
    assert critical


def test_critical_penalty_and_floor():
    tests = [
        _t("1", True, 12, 12),
        _t("8", False, 6, 16, critical_hallucination=True, invented_apis=["session.atomic()"]),
    ]
    run = build_run_result(
        model="x",
        ollama_version="1",
        context="8K",
        mode="full",
        tests=tests,
    )
    assert run.critical_hallucination
    assert run.penalty == 15
    assert run.total == max(0, 12 + 6 - 15)


def test_false_premise_not_in_hundred():
    tests = [
        _t("1", True, 12, 12),
        _t("false_premise", True, 10, 10, scored=False),
    ]
    run = build_run_result(
        model="x", ollama_version="1", context="8K", mode="knowledge", tests=tests
    )
    assert run.total == 12
    assert run.total_max == 12
    assert run.dimensions["hallucination_resistance"] == 10.0


def test_verdict_only_full():
    assert verdict_for(92, "full") == "EXCELLENT CODING AGENT"
    assert verdict_for(92, "knowledge") is None
    assert verdict_for(10, "agent") is None


def test_case_a_real_api_is_not_critical_hallucination():
    points, critical = score_test8(
        contradiction=False,
        real_api=True,
        tests_pass=True,
        invented_apis=[],
    )
    assert points == 10.0
    assert not critical
    run = build_run_result(
        model="x",
        ollama_version="1",
        context="8K",
        mode="agent",
        tests=[
            _t("8", False, points, 16, checks={
                "contradiction": False,
                "real_api": True,
                "tests": True,
                "illegal": [],
            }),
        ],
    )
    assert run.critical_hallucination is False
    assert run.penalty == 0.0
    assert run.invented_apis == []
    assert run.total == 10.0
    assert run.dimensions["hallucination_resistance"] == 10.0 * 10.0 / 16.0


def test_case_b_invented_api_sets_penalty():
    points, critical = score_test8(
        contradiction=False,
        real_api=False,
        tests_pass=False,
        invented_apis=["session.atomic()"],
    )
    assert points == 0.0
    assert critical
    run = build_run_result(
        model="x",
        ollama_version="1",
        context="8K",
        mode="agent",
        tests=[
            _t(
                "8",
                False,
                points,
                16,
                critical_hallucination=True,
                invented_apis=["session.atomic()"],
                checks={"illegal": []},
            ),
        ],
    )
    assert run.critical_hallucination is True
    assert run.penalty == 15.0
    assert run.invented_apis == ["session.atomic()"]
    assert run.total == 0.0


def test_qwen_like_totals_and_dimension_split():
    """Scope fail is test 6; pytest fail is test 5; test 8 Case A is not invented."""
    tests = [
        _t("5", False, 0, 12, checks={"pytest": False, "illegal": ["tests/test_app.py"]}),
        _t("6", False, 0, 12, checks={"illegal": ["tests/test_app.py"]}),
        _t("7", False, 0, 12, checks={"pytest": False, "illegal": []}),
        _t(
            "8",
            False,
            10,
            16,
            checks={
                "contradiction": False,
                "real_api": True,
                "tests": True,
                "illegal": ["app/models.py"],
            },
        ),
    ]
    run = build_run_result(
        model="qwen2.5-coder:14b-instruct",
        ollama_version="1",
        context="32K",
        mode="agent",
        tests=tests,
    )
    assert run.total == 10.0
    assert run.total_max == 52.0
    assert run.penalty == 0.0
    assert run.verdict is None
    assert run.critical_hallucination is False
    assert run.dimensions["instruction_following"] == 0.0
    assert run.dimensions["self_correction"] == 0.0
    assert run.dimensions["hallucination_resistance"] == 10.0 * 10.0 / 16.0
    # Diff-Disziplin of 6 and 8, not test-8 trap points (would have been 3.125).
    assert run.dimensions["agent_discipline"] == 0.0


def test_agent_discipline_ignores_test8_trap_points_when_diff_is_clean():
    tests = [
        _t("6", True, 12, 12, checks={"illegal": []}),
        _t("8", False, 0, 16, checks={"illegal": []}),
    ]
    run = build_run_result(
        model="x", ollama_version="1", context="8K", mode="agent", tests=tests
    )
    assert run.dimensions["agent_discipline"] == 10.0
    assert run.dimensions["hallucination_resistance"] == 0.0
