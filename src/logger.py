"""
logger.py — Structured JSONL trace logger for AgentOps / LLMOps observability.
Every agent call writes one record per stage to logs/agent_trace.jsonl.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.config import TRACE_FILE


def log_event(agent_name: str, stage: str, payload: dict[str, Any]) -> None:
    """
    Append a structured JSON trace record to the JSONL log file.

    Args:
        agent_name: Name of the agent or component producing the log.
        stage:      Execution stage label (e.g. 'start', 'end', 'tool_call_*').
        payload:    Arbitrary JSON-serialisable metadata dict.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent":     agent_name,
        "stage":     stage,
        "payload":   payload,
    }
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
