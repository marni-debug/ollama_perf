from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from llm_fitness.stage1.graders import (
    Grade,
    grade_api_hallucination,
    grade_asyncio,
    grade_error_detection,
    grade_false_premise,
    grade_late_binding,
)


@dataclass(frozen=True)
class KnowledgeCase:
    id: str
    name: str
    prompt: str
    grader: Callable[[str], Grade]
    scored: bool = True


SCORED_CASES: list[KnowledgeCase] = [
    KnowledgeCase(
        id="1",
        name="Python API Hallucination",
        grader=grade_api_hallucination,
        prompt="""Kann ich das so mergen? Ein Kollege hat diesen Helper geschrieben:

```python
from pathlib import Path

def prepare_output(path: str) -> Path:
    p = Path(path)
    p.ensure_dir()
    return p
```

Wir sind auf Python 3.11. Funktioniert das? Falls ja, okay. Falls nicht, sag mir was kaputt ist und wie es richtig geht.""",
    ),
    KnowledgeCase(
        id="2",
        name="Python Semantic Reasoning",
        grader=grade_late_binding,
        prompt="""Was gibt dieser Schnipsel aus und warum?

```python
fns = []
for i in range(3):
    fns.append(lambda: i)
print(fns[0](), fns[1](), fns[2]())
```

Nur das Laufzeitverhalten, bitte keine Stil-Diskussion.""",
    ),
    KnowledgeCase(
        id="3",
        name="Asyncio Semantics",
        grader=grade_asyncio,
        prompt="""Review bitte:

```python
import asyncio

async def fetch(n):
    return n

async def main():
    results = asyncio.gather(fetch(1), fetch(2))
    return results
```

Ist `results` am Ende `[1, 2]`? Wenn nicht, was passiert wirklich?""",
    ),
    KnowledgeCase(
        id="4",
        name="Error Detection",
        grader=grade_error_detection,
        prompt="""Finde den funktionalen Bug. Style ist egal.

```python
def load_tags(tag, cache={}):
    if tag not in cache:
        cache[tag] = []
    cache[tag].append(tag)
    return cache
```

Wird in einem Web-Worker mehrfach aufgerufen. Was geht schief?""",
    ),
]

FALSE_PREMISE = KnowledgeCase(
    id="false_premise",
    name="False Premise Detection",
    scored=False,
    grader=grade_false_premise,
    prompt="""Kurze Korrektur für unser Snippet. Erinnerung: In Python 3.11 ist `x or []` immer `True` oder `False` (Boolean), nie die Liste selbst. Bitte `default_items` so umschreiben, dass wir diese Boolean-Semantik nutzen und trotzdem eine Liste von Items zurückgeben.

```python
def default_items(x):
    return x or []
```

Wenn an der Erinnerung etwas nicht stimmt, sag das zuerst.""",
)


def knowledge_cases() -> list[KnowledgeCase]:
    return [*SCORED_CASES, FALSE_PREMISE]
