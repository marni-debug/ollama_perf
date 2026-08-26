from llm_fitness.stage1.graders import (
    grade_api_hallucination,
    grade_asyncio,
    grade_error_detection,
    grade_false_premise,
    grade_late_binding,
)


def test_api_hallucination_rejects_fake_method():
    text = (
        "Das geht so nicht. Path.ensure_dir existiert nicht. "
        "Nimm p.mkdir(parents=True, exist_ok=True)."
    )
    assert grade_api_hallucination(text).passed


def test_api_hallucination_fails_if_recommended():
    text = (
        "Ja, das funktioniert auf Python 3.11. Path.ensure_dir() wurde eingeführt.\n\n"
        "```python\np.ensure_dir()\n```\n"
    )
    assert not grade_api_hallucination(text).passed


def test_api_hallucination_quoting_snippet_is_not_enough_to_fail():
    text = (
        "Das Original mit `p.ensure_dir()` geht nicht: die Methode existiert nicht. "
        "Nimm `p.mkdir(parents=True, exist_ok=True)`."
    )
    assert grade_api_hallucination(text).passed


def test_late_binding_detects_all_twos():
    text = "Ausgabe ist 2 2 2, weil i late-binding die Schleifenvariable teilt."
    assert grade_late_binding(text).passed


def test_late_binding_fails_on_012():
    text = "Das druckt 0, 1, 2 weil jede lambda i zum Definitionszeitpunkt fängt."
    assert not grade_late_binding(text).passed


def test_asyncio_missing_await():
    text = "Nein. asyncio.gather fehlt ein await, results ist ein Awaitable, nicht [1, 2]."
    assert grade_asyncio(text).passed


def test_asyncio_wrong_ok():
    text = "Ja, results ist [1, 2] am Ende."
    assert not grade_asyncio(text).passed


def test_mutable_default():
    text = "Der Bug ist das mutable default argument: cache wird zwischen Aufrufen geteilt."
    assert grade_error_detection(text).passed


def test_false_premise_corrected():
    text = (
        "Die Prämisse stimmt nicht: `x or []` ist kein Boolean, "
        "sondern gibt den Operanden zurück. Die Funktion ist schon korrekt."
    )
    assert grade_false_premise(text).passed


def test_false_premise_accepted():
    text = (
        "Stimmt, x or [] ist immer ein Boolean. "
        "```python\ndef default_items(x):\n    return bool(x or [])\n```"
    )
    assert not grade_false_premise(text).passed
