import hashlib

import pytest

from compiler.analyzer import CATEGORIES
from compiler.rules import INVENTED_API_INSTRUCTION, load_rules
from compiler.templates import NO_EXTRA_INSTRUCTIONS
from test_prompt_compiler import compile_tmp, failed_test, make_fitness, write_rules_dir

SHIPPED = load_rules()
CATEGORY_INSTRUCTIONS = {name: SHIPPED.categories[name].instructions for name in CATEGORIES}
FIRST_INSTRUCTION = {name: instructions[0] for name, instructions in CATEGORY_INSTRUCTIONS.items()}

TASK_MARKER = "TASK-MARKER-ORANGE-PELIKAN-9173"
DEDUP_MARKER = "USE-ONLY-DECLARED-APIS-TEST-MARKER"

STRONG = {name: 10.0 for name in CATEGORIES}
PROFILE_TASK = "Implement the health endpoint as specified.\n"


def _fitness_dims(target, raw, **overrides):
    dims = dict(STRONG)
    dims[target] = raw
    return make_fitness(dimensions=dims, **overrides)


def _task_section(prompt: str) -> str:
    start = prompt.index("TASK\n\n") + len("TASK\n\n")
    end = prompt.index("\nREQUIREMENT HANDLING\n")
    return prompt[start:end]


def _assert_absent(prompt: str, instructions):
    for instr in instructions:
        assert instr not in prompt


def test_profile_a_vs_b_same_task_hallucination_rules(tmp_path):
    task = PROFILE_TASK
    dims_a = dict(STRONG)
    dims_a["hallucination_resistance"] = 0.0
    dims_b = dict(STRONG)
    result_a = compile_tmp(tmp_path, make_fitness(dimensions=dims_a), task=task)
    result_b = compile_tmp(tmp_path, make_fitness(dimensions=dims_b), task=task)

    hall_rules = CATEGORY_INSTRUCTIONS["hallucination_resistance"]
    assert result_a.profile.scores["hallucination_resistance"] == pytest.approx(0.0)
    assert result_a.profile.classifications["hallucination_resistance"] == "weakness"
    assert "hallucination_resistance" in result_a.profile.weaknesses
    assert result_a.selected.by_category["hallucination_resistance"] == hall_rules
    for instr in hall_rules:
        assert instr in result_a.prompt
        assert result_a.prompt.count(instr) == 1
    for name in CATEGORIES:
        if name == "hallucination_resistance":
            continue
        assert result_a.profile.classifications[name] == "strength"
        assert name not in result_a.profile.weaknesses
        assert result_a.selected.by_category[name] == ()
        _assert_absent(result_a.prompt, CATEGORY_INSTRUCTIONS[name])
    assert INVENTED_API_INSTRUCTION not in result_a.prompt

    assert result_b.profile.scores["hallucination_resistance"] == pytest.approx(1.0)
    assert result_b.profile.classifications["hallucination_resistance"] == "strength"
    assert "hallucination_resistance" not in result_b.profile.weaknesses
    assert result_b.selected.by_category["hallucination_resistance"] == ()
    assert result_b.selected.operating == ()
    assert result_b.selected.extras == ()
    _assert_absent(result_b.prompt, hall_rules)
    _assert_absent(result_b.prompt, [instr for name in CATEGORIES for instr in CATEGORY_INSTRUCTIONS[name]])
    assert INVENTED_API_INSTRUCTION not in result_b.prompt
    assert NO_EXTRA_INSTRUCTIONS in result_b.prompt
    assert result_b.profile.weaknesses == []

    for instr in hall_rules:
        assert instr in result_a.prompt
        assert instr not in result_b.prompt
    assert result_a.prompt != result_b.prompt


@pytest.mark.parametrize("dimension", list(CATEGORIES))
@pytest.mark.parametrize(
    ("raw", "expect_rule"),
    [
        (0.0, True),
        (6.5, False),
        (10.0, False),
    ],
)
def test_dimension_yaml_rule_only_when_weak(tmp_path, dimension, raw, expect_rule):
    result = compile_tmp(tmp_path, _fitness_dims(dimension, raw))
    first = FIRST_INSTRUCTION[dimension]
    rules = CATEGORY_INSTRUCTIONS[dimension]
    if expect_rule:
        assert result.profile.classifications[dimension] == "weakness"
        assert result.selected.by_category[dimension] == rules
        assert first in result.prompt
        for instr in rules:
            assert instr in result.prompt
            assert result.prompt.count(instr) == 1
        for other in CATEGORIES:
            if other == dimension:
                continue
            _assert_absent(result.prompt, CATEGORY_INSTRUCTIONS[other])
            assert result.selected.by_category[other] == ()
    else:
        assert result.profile.classifications[dimension] != "weakness"
        assert result.selected.by_category[dimension] == ()
        _assert_absent(result.prompt, rules)


def test_null_code_reasoning_omitted_from_profile_and_prompt(tmp_path):
    data = make_fitness(dimensions={"code_reasoning": None})
    result = compile_tmp(tmp_path, data)
    assert "code_reasoning" not in result.profile.scores
    assert "code_reasoning" not in result.profile.classifications
    assert "code_reasoning" not in result.profile.strengths
    assert "code_reasoning" not in result.profile.weaknesses
    assert set(result.profile.scores) == set(CATEGORIES) - {"code_reasoning"}
    assert result.selected.by_category["code_reasoning"] == ()
    _assert_absent(result.prompt, CATEGORY_INSTRUCTIONS["code_reasoning"])
    dumped = result.profile.to_dict()
    assert "code_reasoning" not in dumped["scores"]
    assert "code_reasoning" not in dumped["classifications"]


def test_task_section_preserves_original_text_and_marker(tmp_path):
    task = (
        "Complete the following assignment.\n"
        f"{TASK_MARKER}\n"
        "Keep punctuation, spacing,  and this paragraph.\n"
    )
    result = compile_tmp(tmp_path, make_fitness(), task=task)
    embedded = _task_section(result.prompt)
    assert embedded == task
    assert TASK_MARKER in embedded
    assert TASK_MARKER in result.prompt
    assert result.prompt.count(TASK_MARKER) == 1


def test_global_dedup_shipped_instruction_appears_once(tmp_path):
    hall = FIRST_INSTRUCTION["hallucination_resistance"]
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"hallucination_resistance": 0.0},
            tests=[failed_test("8")],
        ),
    )
    assert hall in result.prompt
    assert result.prompt.count(hall) == 1
    for instr in CATEGORY_INSTRUCTIONS["hallucination_resistance"]:
        assert result.prompt.count(instr) == 1


def test_global_dedup_injected_marker_appears_once(tmp_path):
    rules_dir = write_rules_dir(
        tmp_path,
        extra_files={
            "hallucination_resistance.yaml": (
                'description: "hall"\n'
                f'instructions:\n  - "{DEDUP_MARKER}"\n'
            ),
            "instruction_following.yaml": (
                'description: "instr"\n'
                f'instructions:\n  - "{DEDUP_MARKER}"\n'
            ),
        },
    )
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={
                "hallucination_resistance": 0.0,
                "instruction_following": 0.0,
            }
        ),
        rules_dir=rules_dir,
    )
    assert DEDUP_MARKER in result.prompt
    assert result.prompt.count(DEDUP_MARKER) == 1


def test_compile_three_times_deterministic_sha256(tmp_path):
    data = make_fitness(
        dimensions={
            "hallucination_resistance": 0.0,
            "code_reasoning": 6.5,
            "instruction_following": 10.0,
            "agent_discipline": 5.0,
            "self_correction": 10.0,
        },
        tests=[failed_test("8"), failed_test("6")],
        invented_apis=["session.atomic()"],
    )
    task = "Add a health endpoint.\nKeep this wording.\n"
    r1 = compile_tmp(tmp_path, data, task=task)
    r2 = compile_tmp(tmp_path, data, task=task)
    r3 = compile_tmp(tmp_path, data, task=task)
    assert r1.prompt == r2.prompt == r3.prompt
    digest = hashlib.sha256(r1.prompt.encode("utf-8")).hexdigest()
    assert hashlib.sha256(r2.prompt.encode("utf-8")).hexdigest() == digest
    assert hashlib.sha256(r3.prompt.encode("utf-8")).hexdigest() == digest
    assert r1.profile.to_dict() == r2.profile.to_dict() == r3.profile.to_dict()
