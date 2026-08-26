import json
from pathlib import Path

from llm_fitness.ollama_client import NUM_CTX
from llm_fitness.persist import load_run, save_run
from llm_fitness.report import render, render_compare
from llm_fitness.results import TestResult
from llm_fitness.scoring import build_run_result


def test_roundtrip_and_render(tmp_path: Path):
    tests = [
        TestResult("1", "Python API Hallucination", 1, True, 12, 12, scored=True),
        TestResult("2", "Python Semantic Reasoning", 1, False, 0, 12, detail="missed", scored=True),
    ]
    run = build_run_result(
        model="ornith:35b",
        ollama_version="0.32.15",
        context="256K",
        mode="knowledge",
        tests=tests,
    )
    path = save_run(run, directory=tmp_path)
    loaded = load_run(path)
    assert loaded.model == "ornith:35b"
    assert loaded.total == 12
    text = render(loaded)
    assert "LLM CODING AGENT FITNESS TEST" in text
    assert "PASS" in text
    assert "FAIL" in text
    table = render_compare([loaded])
    assert "ORNITH35B" in table


def test_saved_report_uses_effective_num_ctx(tmp_path: Path):
    tests = [TestResult("1", "Python API Hallucination", 1, True, 12, 12, scored=True)]
    run = build_run_result(
        model="qwen3:14b",
        ollama_version="0.32.15",
        context="8K",
        mode="knowledge",
        tests=tests,
    )
    run.context_effective = NUM_CTX
    run.context_model_max = 40960
    path = save_run(run, directory=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["context"]["effective"] == NUM_CTX
    assert payload["context"]["model_max"] == 40960
    loaded = load_run(path)
    assert loaded.context_effective == NUM_CTX
    assert loaded.context_model_max == 40960
    assert loaded.context == "8K"
