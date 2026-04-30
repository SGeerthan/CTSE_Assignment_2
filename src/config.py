"""
config.py — Central configuration for the MAS project.
All paths are relative to the project root so the project
can be placed anywhere on your local machine.
"""
from pathlib import Path

# ── Root of the project (one level above src/) ──────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path   = BASE_DIR / "data"
LOG_DIR: Path    = BASE_DIR / "logs"
REPORT_DIR: Path = BASE_DIR / "outputs"

DB_PATH: Path        = DATA_DIR / "bank_support.db"
TRACE_FILE: Path     = LOG_DIR  / "agent_trace.jsonl"

# ── LLM settings ─────────────────────────────────────────────────────────────
OLLAMA_MODEL: str    = "qwen2.5:3b"
OLLAMA_TEMPERATURE: float = 0.1

# ── Create dirs on import ─────────────────────────────────────────────────────
for _d in [DATA_DIR, LOG_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
