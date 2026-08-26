from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
CACHE_DIR = Path.home() / ".cache" / "llm-fitness"
RUNS_DIR = CACHE_DIR / "runs"
VENV_CACHE_DIR = CACHE_DIR / "venvs"
RESULTS_DIR = REPO_ROOT / "results"
