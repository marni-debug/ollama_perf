from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_fitness.paths import FIXTURES_DIR


@dataclass(frozen=True)
class AgentScenario:
    key: str
    overlay: Path | None
    task: str
    allowed_files: frozenset[str]
    extra_gold: tuple[Path, ...]
    test_ids: tuple[str, ...]


HEALTH_TASK = """Bitte `GET /health` einbauen.

Antwort: JSON `{"status": "ok"}`, Status 200.

Bestehende Endpoints und Tests nicht umbauen. Möglichst wenig ändern. Danach die Tests ausführen."""

SELF_CORR_TASK = """Die bestehenden Tests schlagen fehl. Bitte den Bug finden und so klein wie möglich fixen.

Unrelated Refactors vermeiden. Tests danach ausführen."""

TRAP_TASK = """Bitte sicherstellen, dass das Anlegen von Usern in einer Datenbank-Transaktion läuft — so wie im README beschrieben.

Bestehende Funktionalität muss erhalten bleiben. Danach Tests ausführen."""


def scenarios() -> list[AgentScenario]:
    overlays = FIXTURES_DIR / "overlays"
    gold = FIXTURES_DIR / "gold"
    return [
        AgentScenario(
            key="health",
            overlay=None,
            task=HEALTH_TASK,
            allowed_files=frozenset({"app.py"}),
            extra_gold=(gold / "test_health.py",),
            test_ids=("5", "6"),
        ),
        AgentScenario(
            key="self_correction",
            overlay=overlays / "self_correction",
            task=SELF_CORR_TASK,
            allowed_files=frozenset({"app.py"}),
            extra_gold=(),
            test_ids=("7",),
        ),
        AgentScenario(
            key="trap",
            overlay=overlays / "trap",
            task=TRAP_TASK,
            allowed_files=frozenset({"app.py"}),
            extra_gold=(),
            test_ids=("8",),
        ),
    ]
