# Anwendungsleitfaden

Praktische Anleitung für die Werkzeuge in diesem Repository. Ziel ist nicht ein öffentlicher Leaderboard-Score, sondern die Frage:

> Welches lokale Ollama-Modell eignet sich auf **dieser** Hardware als Coding-Agent — und hilft ein aus dem Fitness-Lauf gebauter Prompt?

Voraussetzung: Python 3.11+, laufendes Ollama (`http://127.0.0.1:11434`), das gewünschte Modell ist geladen (`ollama pull …` / `ollama list`).

Keine extra Python-Pakete. Tests: `python -m pytest tests -q`.

---

## 1. Überblick

Drei getrennte Werkzeuge, eine Kette:

```text
1. Fitness-Check     python check_llm.py -m MODEL
                     → results/<model>-<zeit>.json

2. Prompt Compiler   python prompt_compiler.py --fitness … --task …
                     → compiled.md  (kein LLM-Aufruf)

3. A/B-Experiment    python prompt_compiler.py experiment …
                     → experiment.json
                     Baseline-Prompt vs. Compiled Prompt, gleicher Holdout
```

`ollama_perf_logger.py` ist unabhängig (Journalctl-/Durchsatz-Logging) und gehört nicht in diese Kette.

Der Compiler erzeugt den Prompt **nicht** neu, während das Experiment läuft. Ablauf bewusst nacheinander: erst messen, dann kompilieren, dann vergleichen.

---

## 2. Fitness-Check

Misst, wie gut ein Modell als **autonomer Coding-Agent** (Cline-ähnliche XML-Tools) taugt. Das ist kein HumanEval-Klon.

### Aufrufe

```bash
# Vollständig: Stufe 1 (Wissen) + Stufe 2 (Agent im Temp-Workspace)
python check_llm.py -m ornith:35b

# Nur Prompt-Wissen (schnell, kein Workspace)
python check_llm.py -m qwen3:14b --mode knowledge

# Nur Agent-Szenarien
python check_llm.py -m ornith:35b --mode agent --max-turns 5

# Mehrere Modelle nacheinander, danach Tabelle
python check_llm.py -m ornith:35b -m qwen3:14b --mode knowledge
python check_llm.py --compare results/ornith-35b-*.json results/qwen3-14b-*.json
```

Ollama muss laufen. Ergebnisse landen in `results/` (gitignoriert).

### Modi

| `--mode` | Was läuft | Report |
|----------|-----------|--------|
| `full` (Default) | 8 Tests (4 Wissen + 4 Agent-Scores) | Score `/100`, Verdict |
| `knowledge` | Tests 1–4 plus optionaler False-Premise-Fall | Score `/48`, kein Verdict |
| `agent` | Tests 5–8 in drei Workspaces | Score `/52`, kein Verdict |

`--max-turns` gilt nur für den Agent-Loop (Default 20). `--host` ändert die Ollama-URL.

### Was der Report bedeutet

- **Context:** effektives Fenster (`8K` = `num_ctx=8192`), daneben das Modellmaximum. Pro Antwort höchstens `num_predict=4096` Tokens (inkl. Thinking).
- **Dimensionen** (0–10, `n/a` wenn der Modus die Stufe nicht enthält):

  | Dimension | Inhalt |
  |-----------|--------|
  | Hallucination resistance | Fake-APIs, README-Fallen |
  | Code reasoning | Semantik, asyncio, Bugs |
  | Instruction following | Feature wie spezifiziert (Test 5) |
  | Agent discipline | nur erlaubte Dateien (Diff) |
  | Self-correction | bestehenden Fail fixen |

- **Critical Hallucination (−15):** erfundene Session-API (z. B. `db.session.atomic()`), unabhängig von `atomic()` vs. anderer Fake-API.
- **temperature=0** verringert Streuung, macht Läufe **nicht** bitidentisch.

### Beispiel: JSON nutzen

```bash
ls results/ornith-35b-*.json
```

Relevant für den Compiler:

- `model`, `mode`, `dimensions` (Werte 0–10 oder `null`)
- `tests[]` mit `passed`, `points`, `detail`
- `critical_hallucination`, `invented_apis`

`null` bei einer Dimension heißt: in diesem Lauf nicht gemessen (z. B. `code_reasoning` im reinen `--mode agent`). Der Compiler lässt sie weg, er rät nicht.

---

## 3. Prompt Compiler

Liest ein Fitness-JSON und eine **unveränderte** Task-Datei. Kein Ollama, kein Zufall. Gleiche Eingaben → gleiche Bytes.

Schwellen (zentral in `rules/thresholds.yaml`): Score = Dimension/10. `≥ 0.75` Stärke, `< 0.60` Schwäche, dazwischen neutral. Nur **Schwächen** bekommen Kompensationsregeln aus `rules/`. Stärken stehen im Profil, ohne Extra-Instruktionen.

### Aufruf

```bash
python prompt_compiler.py \
    --fitness results/ornith-35b-20260826T223730Z.json \
    --task task.md \
    --output compiled.md \
    --profile \
    --explain
```

| Flag | Wirkung |
|------|---------|
| `--fitness` | Fitness-JSON (Pflicht) |
| `--task` | Task-Datei, wird 1:1 in die Sektion `TASK` kopiert (Pflicht) |
| `--output` | Prompt in Datei; ohne Flag: Prompt auf stdout |
| `--profile` | Capability-Profil als JSON **auf stdout** (kein Dateipfad) |
| `--explain` | Pro Dimension: Score, Klassifikation, angewandte Regeln (stdout) |

`--profile compiled.json` gibt es nicht. Profil umleiten:

```bash
python prompt_compiler.py \
    --fitness results/ornith-35b-20260826T223730Z.json \
    --task task.md \
    --output compiled.md \
    --profile \
    > profile.json
```

Dann enthält `profile.json` nur das Profil, `compiled.md` den Prompt.

### Task-Datei

Der Compiler fasst den Task nicht an. Beispiel `task.md`:

```markdown
Implementiere GET /health.

Antwort: JSON {"status":"ok"}, Status 200.
Bestehende Endpoints nicht ändern. Danach pytest ausführen.
```

### Prompt-Aufbau

Feste Überschriften in dieser Reihenfolge: `ROLE`, `MODEL-SPECIFIC OPERATING PROFILE`, `TASK`, `REQUIREMENT HANDLING`, `IMPLEMENTATION GUIDANCE`, `VERIFICATION`, `COMPLETION CRITERIA`.

Identische Regeltexte erscheinen im fertigen Prompt nur **einmal**.

### Fehler

Ungültiges JSON, fehlendes `model` oder fehlendes `dimensions`-Objekt: Meldung auf stderr, Exit 1, **keine** (auch nicht teilweise) `--output`-Datei.

---

## 4. A/B-Experiment

Vergleicht **denselben** Holdout unter zwei System-Prompts bei **kontrollierten A/B-Bedingungen**. Die Experimentdefinition ist deterministisch (Datei-Hashes, Task-Paarung, Knöpfe). Der LLM-Lauf selbst ist **nicht** deterministisch.

```text
Evaluation A:  Modell + Holdout-Task + baseline.md  + eigenes Temp-Workspace
Evaluation B:  Modell + Holdout-Task + compiled.md  + eigenes Temp-Workspace
```

Reihenfolge pro Task: erst A, dann B. Dieselbe `task_id` und derselbe `input` auf beiden Seiten. Der Runner kompiliert **nicht** selbst.

### Drei Schichten, nicht vermischen

| Schicht | Rolle |
|---------|--------|
| Fitness (`check_llm.py`) | Echte Agent-Bewertung (Tests, Diff, Halluzinationen) |
| Compiler | Prompt-Bytes aus Fitness-JSON + Task. Gleiche Inputs → gleiche Bytes |
| A/B-Experiment | Zwei fertige Prompts, gleiches Modell, gleicher Holdout, gleiche Knöpfe |

Ein A/B-Delta ist **kein** Fitness-Delta. Der Compiler ist byte-deterministisch; die LLM-Antworten sind es nicht.

### Live-Default ist ein Rauchtest

`evaluator: "non_empty_response"`. Ein Ollama-Chat (`system=Prompt`, `user=Task-Input`) gilt als bestanden, wenn die Antwort nicht leer ist: `passed = bool(content.strip())`. Keine Tools. Das Workspace dient nur der Isolation; der Default-Evaluator ignoriert es.

Das ist **kein** Code-Qualitätsmaß und **kein** Ersatz für die Fitness-Tests.

- `improved` heißt nicht „Task gelöst“.
- 5/10 gegen 7/10 nicht-leere Antworten ist **nicht** „+20 % Codequalität“.
- `delta.passed == 2` heißt: zwei Tasks mehr mit nicht-leerer Antwort, nicht zwei Tasks mehr korrekt.

Echte Coding-Qualität weiter mit `check_llm.py` messen. Ein späterer deterministischer Task-Evaluator (pytest, Diffs, Invarianten) ist nicht implementiert.

### Holdout-JSON

```json
{
  "name": "holdout",
  "tasks": [
    {
      "id": "health_001",
      "input": "Implementiere GET /health. JSON {\"status\":\"ok\"}, HTTP 200. /users nicht ändern. pytest ausführen."
    },
    {
      "id": "bugfix_002",
      "input": "Die bestehenden Tests sind rot. Finde den Bug und ändere so wenig wie möglich."
    }
  ]
}
```

`name` und `tasks` sind Pflicht. Jeder Task braucht eindeutiges `id` und `input`. Der `input` wird als User-Nachricht verwendet, nicht umgeschrieben.

### Baseline-Prompt

Eigene Datei, z. B. `baseline.md` — das, was der Agent **ohne** Compiler bekäme:

```text
You are an autonomous coding agent.
Prefer small edits. Run tests after behavior changes.
Do not invent APIs that are not in the project.
```

`compiled.md` kommt unverändert aus Schritt 3.

### Aufruf

```bash
python prompt_compiler.py experiment \
    --model ornith:35b \
    --benchmark holdout.json \
    --baseline-prompt baseline.md \
    --compiled-prompt compiled.md \
    --output experiment.json
```

Hashes sind SHA-256 über die **rohen Datei-Bytes** (kein `strip()`, keine Normalisierung). Ein zusätzliches Leerzeichen ändert den Hash.

Pro Task-Arm ein eigenes Temp-Verzeichnis; A und B teilen keinen Pfad. Nach dem Lauf werden die Verzeichnisse gelöscht.

### Ergebnis lesen

- `evaluator`: immer `non_empty_response` (Live-Default)
- `baseline` / `compiled`: Summe der Task-Scores, Anzahl passed/failed, `prompt_sha256`
- `delta.score` / `delta.passed`: compiled − baseline (Rauchtest-Zählung, keine Qualitätsdifferenz)
- `tasks[].comparison`: `improved` (fail→pass), `regressed` (pass→fail), `unchanged` — bezogen auf nicht-leere Antwort

Gleiche Knöpfe für A und B: `temperature=0`, `num_ctx=8192`, `num_predict=4096`. Ollama hat keinen Seed; `temperature=0` friert Sampling nicht ein. Das steht in `notes` der JSON.

---

## 5. Typische End-to-End-Session

```bash
# 1) Modell messen (Agent reicht für den Compiler; full ist gründlicher)
python check_llm.py -m ornith:35b --mode agent --max-turns 8

# 2) Neueste JSON merken
ls -t results/ornith-35b-*.json | head -1

# 3) Prompt bauen (Task = echte Arbeit, nicht der Fitness-Prompt)
python prompt_compiler.py \
    --fitness results/ornith-35b-YYYYMMDDThhmmssZ.json \
    --task meine-aufgabe.md \
    --output compiled.md \
    --explain

# 4) Optional: Holdout gegen Baseline
python prompt_compiler.py experiment \
    --model ornith:35b \
    --benchmark holdout.json \
    --baseline-prompt baseline.md \
    --compiled-prompt compiled.md \
    --output experiment.json
```

Compiler und Experiment brauchen **kein** zweites Fitness-JSON. Den Compiler nicht mit dem Holdout füttern, den Holdout nicht durch den Compiler jagen.

---

## Tipps & Tricks

- **`--mode knowledge` zum Vorfiltern.** Agent-Läufe (35B, viele Turns) dauern. Erst Wissen, dann Agent für die Kandidaten.
- **`--max-turns` bewusst klein.** Für Rauchtests 3–5; der Fitness-Alltag darf 20 bleiben. Ein Turn kann bis 4096 Tokens ziehen.
- **Immer `--output` beim Kompilieren.** Sonst landet der Prompt auf stdout und vermischt sich mit `--profile`/`--explain`.
- **Task-Datei = Produktionsauftrag.** Fitness-Fälle nicht als `--task` recyceln, sonst misst ihr denselben Stoff zweimal und der Compiler „kennt“ die Prüfung.
- **Holdout klein und stabil halten.** Gleiche Task-IDs und -Reihenfolge, sonst sind Deltas nicht vergleichbar.
- **Hashes prüfen**, wenn ihr unsicher seid, welche Prompt-Datei im Experiment war: `sha256sum baseline.md compiled.md` muss zu `conditions.*.prompt_sha256` passen.
- **`results/` nicht committen.** Laufzeitdaten; JSON bei Bedarf gezielt kopieren.
- **Thinking-Modelle (Qwen3).** Tool-XML muss in `content` stehen, nicht nur in `thinking`. Der Fitness-Agent parst nur `content`.
- **Critical Hallucination.** Ein `atomic()`-Fail zieht −15 vom Total. Der Compiler setzt dann typischerweise Halluzinationsregeln — das ist Absicht, kein Bug.
- **Vergleichstabelle** nur über gespeicherte JSON: `python check_llm.py --compare results/*.json`.

---

## FAQ

**Muss Ollama für den Compiler laufen?**  
Nein. Nur `check_llm.py` und das Live-A/B-Experiment sprechen Ollama an.

**Warum zeigt der Report 8K, `ollama show` aber 40K?**  
8K ist das **feste** Request-Fenster (`num_ctx=8192`) für alle Modelle. 40K ist das Modellmaximum. Fairer Vergleich schlägt festes Fenster.

**Warum bricht eine einzelne Antwort nicht bei 8K ab, obwohl früher zehntausende Tokens liefen?**  
`num_ctx` ist das KV-Fenster (Context-Shift möglich). Die Kappe pro Antwort ist `num_predict=4096`.

**`--profile profile.json` funktioniert nicht.**  
`--profile` ist ein Schalter. Datei: `--output` für den Prompt, stdout umleiten fürs Profil.

**Der Task im Prompt weicht vom Original ab.**  
Sollte nicht vorkommen. Compiler-Tests prüfen Bytegleichheit. Wenn doch: Task-Datei Encoding (UTF-8) und keine unsichtbaren Editor-Änderungen.

**`code_reasoning` fehlt im Profil.**  
Beim reinen `--mode agent` ist die Dimension `null`. Der Compiler lässt sie weg und erfindet keine Schwäche.

**Stärken bekommen trotzdem lange Anweisungen.**  
YAML-Kompensation nur bei Schwächen. Neutralzeilen in REQUIREMENT/IMPLEMENTATION/VERIFICATION sind fest und kurz. Halluzinationsregeln können in VERIFICATION stehen, wenn genau diese Dimension schwach ist.

**A/B sagt `improved`, aber der Code ist falsch.**  
Live-Default wertet jede nicht-leere Antwort als pass. Für echte Codequalität weiter `check_llm.py` bzw. einen injizierten Evaluator nutzen.

**Experiment schreibt keine Datei und Exit 1.**  
Typisch: Holdout kein Objekt, `tasks` fehlt, doppelte Task-ID, leere Prompt-Datei, nicht-UTF-8. Meldung steht auf stderr.

**Kann ich zwei Modelle in einem Experiment vergleichen?**  
Nein. Ein Modell, ein Holdout, zwei Prompts. Cross-Model ist nicht implementiert.

**Sind zwei Fitness-Läufe desselben Modells identisch?**  
Nein. `temperature=0` reicht nicht. Compiler-Ausgabe aus **demselben** JSON ist identisch.

**Darf ich auf `main` committen?**  
Nein. Aufgabe → `feat/<name>` von `origin/main` → Pull Request. Merge entscheidest du (`AGENTS.md`).
