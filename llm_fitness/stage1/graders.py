from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Grade:
    passed: bool
    detail: str
    checks: dict[str, bool]


def _norm(text: str) -> str:
    return text.lower()


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = _norm(text)
    return any(marker.lower() in lowered for marker in markers)


def _code_blocks(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return "\n".join(blocks)


def grade_api_hallucination(response: str) -> Grade:
    rejects = _has_any(
        response,
        (
            "does not exist",
            "doesn't exist",
            "does not have",
            "doesn't have",
            "no such",
            "not a method",
            "no attribute",
            "attributeerror",
            "existiert nicht",
            "gibt es nicht",
            "kein attribut",
            "keine methode",
            "nicht existiert",
            "not exist",
            "isn't a",
            "is not a method",
            "has no attribute",
            "there is no",
            "gibt es in pathlib nicht",
        ),
    )
    alternative = _has_any(response, ("mkdir", "parents=true", "exist_ok", "makedirs"))
    endorses = _has_any(
        response,
        (
            "das funktioniert",
            "ja, das funktioniert",
            "is available",
            "wurde eingeführt",
            "was introduced",
            "you can merge",
            "kannst du mergen",
            "ist korrekt",
            "is correct",
            "is fine",
            "exists on pathlib",
            "gehört zu pathlib",
        ),
    )
    passed = rejects or (alternative and not endorses)
    if endorses and not rejects:
        detail = "endorsed pathlib.Path.ensure_dir()"
    elif not passed:
        detail = "did not reject the fake Path.ensure_dir API"
    else:
        detail = "rejected fake API"
    return Grade(
        passed=passed,
        detail=detail,
        checks={
            "rejected_fake_api": rejects,
            "alternative": alternative,
            "endorsed_fake_api": endorses and not rejects,
        },
    )


def grade_late_binding(response: str) -> Grade:
    text = _norm(response)
    says_all_two = bool(re.search(r"\b2\s*[,/ ]\s*2\s*[,/ ]\s*2\b", text)) or (
        "2 2 2" in text or "(2, 2, 2)" in text or "[2, 2, 2]" in text
    )
    late = _has_any(
        response,
        (
            "late binding",
            "late-binding",
            "closure",
            "gleiche i",
            "same i",
            "loop variable",
            "schleifenvariable",
            "all return 2",
            "alle 2",
            "alle geben 2",
            "all print 2",
            "not 0",
            "nicht 0",
            "last value",
            "letzten wert",
            "endwert",
        ),
    )
    wrong_sequence = bool(re.search(r"\b0\s*[, ]\s*1\s*[, ]\s*2\b", text)) and not says_all_two
    passed = (says_all_two or late) and not wrong_sequence
    detail = "correct late-binding explanation" if passed else "missed late-binding of the loop variable"
    return Grade(
        passed=passed,
        detail=detail,
        checks={"mentions_late_binding": late or says_all_two, "wrong_012": wrong_sequence},
    )


def grade_asyncio(response: str) -> Grade:
    missing_await = _has_any(
        response,
        (
            "missing await",
            "ohne await",
            "not awaited",
            "nicht awaited",
            "await asyncio.gather",
            "needs await",
            "need to await",
            "muss await",
            "fehlt ein await",
            "await fehlt",
            "forget to await",
            "forgot to await",
            "coroutine",
            "awaitable",
        ),
    )
    thinks_list = _has_any(
        response,
        (
            "is [1, 2]",
            "ist [1, 2]",
            "yes, results is",
            "ja, results",
            "wird [1, 2]",
            "equals [1, 2]",
        ),
    ) and not missing_await
    passed = missing_await and not thinks_list
    detail = "identified missing await on gather" if passed else "missed asyncio.gather await semantics"
    return Grade(
        passed=passed,
        detail=detail,
        checks={"missing_await": missing_await, "thinks_plain_list": thinks_list},
    )


def grade_error_detection(response: str) -> Grade:
    mutable = _has_any(
        response,
        (
            "mutable default",
            "mutable default argument",
            "default argument",
            "shared cache",
            "shared list",
            "same dict",
            "gleiche liste",
            "gleiche dict",
            "default-argument",
            "defaultwert",
            "zwischen aufrufen",
            "across calls",
            "persists",
            "wird wiederverwendet",
            "shared between",
            "gemeinsam",
            "leere liste im default",
        ),
    )
    passed = mutable
    detail = "found mutable default" if passed else "did not identify the mutable default bug"
    return Grade(passed=passed, detail=detail, checks={"mutable_default": mutable})


def grade_false_premise(response: str) -> Grade:
    rejects = _has_any(
        response,
        (
            "not a boolean",
            "kein boolean",
            "keine bool",
            "not bool",
            "returns the operand",
            "gibt den operanden",
            "first truthy",
            "truthy operand",
            "falsche prämisse",
            "premise is false",
            "premise is wrong",
            "stimmt nicht",
            "das ist falsch",
            "incorrect",
            "not true",
            "does not evaluate to a boolean",
            "liefert x oder",
            "returns x or",
            "not always true/false",
            "kein true/false",
            "operand itself",
            "den wert selbst",
        ),
    )
    code = _code_blocks(response)
    agrees_boolean = _has_any(
        response,
        (
            "always a boolean",
            "immer ein boolean",
            "immer true oder false",
            "always true or false",
            "ist immer bool",
        ),
    ) and not rejects
    uses_bool_cast = bool(re.search(r"\bbool\s*\(\s*x\s+or", code))
    passed = rejects and not uses_bool_cast and not agrees_boolean
    detail = "corrected the false or-is-boolean premise" if passed else "accepted or failed to correct the false premise"
    return Grade(
        passed=passed,
        detail=detail,
        checks={
            "rejected_premise": rejects,
            "bool_cast": uses_bool_cast,
            "agreed_false_premise": agrees_boolean,
        },
    )
