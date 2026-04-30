"""
main.py — Entry point for the Multi-Agent Bank Support Resolver.

Usage:
    python main.py
"""
from __future__ import annotations

import json
import uuid
from pprint import pprint
from urllib.error import URLError
from urllib.request import urlopen

from src.config import OLLAMA_MODEL, TRACE_FILE
from src.database import init_db
from src.models import MASState

DEMO_MESSAGES: list[str] = [
    "Hello, my account ACC001 has a suspicious transaction TXN001. I want to dispute it.",
    "Please block my card linked to account ACC002 immediately.",
    "What is the status of my transaction TXN002 for account ACC002?",
    "I cannot access my account ACC003, can you help me recover it?",
    "I forgot my account number. Can you help with card security?",
]


def run_single(message: str, graph_app) -> MASState:
    """Run one customer message through the full MAS pipeline."""
    state: MASState = {
        "ticket_id":        str(uuid.uuid4()),
        "customer_message": message,
        "errors":           [],
    }
    return graph_app.invoke(state)


def check_ollama_ready() -> tuple[bool, str]:
    """
    Verify local Ollama server is reachable and the configured model exists.
    """
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError:
        return (
            False,
            "Ollama is not reachable at http://127.0.0.1:11434.",
        )
    except Exception as exc:
        return (False, f"Could not validate Ollama status: {exc}")

    model_names = {m.get("name", "") for m in payload.get("models", [])}
    if OLLAMA_MODEL not in model_names:
        return (
            False,
            f"Model '{OLLAMA_MODEL}' was not found in local Ollama models.",
        )
    return (True, "Ollama is ready.")


def print_case_result(result: MASState) -> None:
    """Print a concise result block for one processed message."""
    print(f"  Triage    : {result.get('triage_label')}")
    print(f"  Account ID: {result.get('extracted_account_id')}")
    print(f"  Decision  : {result.get('policy_decision', {}).get('decision_type')} "
          f"(approved={result.get('policy_decision', {}).get('approved')})")
    print(f"  Action    : {result.get('action_result')}")
    print(f"\n  RESPONSE:\n  {result.get('final_response')}")

    if result.get("errors"):
        print(f"\n  [ERRORS]: {result['errors']}")


def run_demo_cases(graph_app) -> None:
    """Run built-in demo messages one by one."""
    for i, msg in enumerate(DEMO_MESSAGES, start=1):
        print(f"\n{'-' * 60}")
        print(f"  CASE {i}: {msg}")
        print("-" * 60)

        try:
            result = run_single(msg, graph_app)
        except Exception as exc:
            print(f"  [runtime-error] Pipeline stopped at CASE {i}: {exc}")
            print("  [hint] Verify local LLM runtime and dependencies, then retry.")
            break

        print_case_result(result)


def run_interactive(graph_app) -> None:
    """Run in interactive mode so users can type their own bank support queries."""
    print("\nType your support message and press Enter.")
    print("Type 'exit' to quit.\n")

    case_no = 1
    while True:
        user_message = input("You> ").strip()
        if not user_message:
            print("Please enter a message (or type 'exit' to quit).")
            continue

        if user_message.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive mode.")
            break

        print(f"\n{'-' * 60}")
        print(f"  CASE {case_no}: {user_message}")
        print("-" * 60)

        try:
            result = run_single(user_message, graph_app)
        except Exception as exc:
            print(f"  [runtime-error] Pipeline stopped: {exc}")
            print("  [hint] Verify local LLM runtime and dependencies, then retry.")
            continue

        print_case_result(result)
        case_no += 1


def main() -> None:
    ready, reason = check_ollama_ready()
    if not ready:
        print(f"\n[startup-check] {reason}")
        print("[startup-check] Run: ollama serve")
        print(f"[startup-check] Then run: ollama pull {OLLAMA_MODEL}")
        return

    # Ensure DB exists and is seeded
    init_db()
    print("[startup-check] Loading workflow...", flush=True)
    from src.graph import app as graph_app
    print("[startup-check] Workflow loaded.", flush=True)

    print("\n" + "=" * 60)
    print("  SE4010 - Multi-Agent Bank Support Resolver")
    print("=" * 60)
    print("  Select mode:")
    print("  1) Interactive input (type your own messages)")
    print("  2) Run predefined demo cases")

    selected = input("\nEnter 1 or 2 (default 1): ").strip()
    if selected == "2":
        run_demo_cases(graph_app)
    else:
        run_interactive(graph_app)

    print(f"\n{'=' * 60}")
    print(f"  Trace log -> {TRACE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
