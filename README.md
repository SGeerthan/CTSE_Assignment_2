# SE4010 — CTSE Assignment 2
## Multi-Agent E-Commerce Support Resolver
**LangGraph · Ollama (qwen2.5:3b) · SQLite · Python 3.12**

---

## Project Structure

```
mas_project/
├── main.py                  ← Run the full demo
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py            ← All paths & settings
│   ├── database.py          ← SQLite setup + seed data
│   ├── logger.py            ← Structured JSONL trace logger
│   ├── models.py            ← Pydantic models + MASState
│   ├── tools.py             ← 5 custom Python tools
│   ├── agents.py            ← 4 agents (triage / order / policy / response)
│   └── graph.py             ← LangGraph workflow + smart router
│
├── tests/
│   ├── __init__.py
│   └── test_mas.py          ← Unit + integration + LLM-as-a-Judge tests
│
├── data/                    ← ecommerce.db  (auto-created on first run)
├── logs/                    ← agent_trace.jsonl  (written at runtime)
└── outputs/                 ← architecture_notes.md  (written at runtime)
```

---

## System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python      | **3.12** ✅ (fully supported) |
| RAM         | 8 GB (16 GB recommended for qwen2.5:3b) |
| Disk        | ~3 GB for the Ollama model |
| OS          | Windows 10/11, macOS 12+, Ubuntu 20.04+ |

---

## Step-by-Step Installation

### Step 1 — Install Ollama

**Windows:**
1. Go to https://ollama.com/download
2. Download the Windows installer and run it
3. Ollama starts automatically as a background service

**macOS:**
```bash
# Option A — download from https://ollama.com/download
# Option B — Homebrew
brew install ollama
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify Ollama is running:
```bash
ollama --version
```

---

### Step 2 — Pull the LLM model

```bash
ollama pull qwen2.5:3b
```

This downloads ~2 GB. Wait until it completes.

Verify:
```bash
ollama list
# Should show: qwen2.5:3b
```

---

### Step 3 — Clone / place the project

Place the `mas_project/` folder anywhere on your PC, for example:
```
C:\Users\YourName\Documents\mas_project\     (Windows)
~/Documents/mas_project/                      (macOS/Linux)
```

---

### Step 4 — Create a Python 3.12 virtual environment

**Windows (Command Prompt or PowerShell):**
```cmd
cd C:\Users\YourName\Documents\mas_project
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
cd ~/Documents/mas_project
python3.12 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

---

### Step 5 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `langgraph`, `langchain`, `langchain-ollama`, `pydantic`, `pytest`.

---

### Step 6 — Initialise the database

```bash
python -m src.database
```

Expected output:
```
[database] Seeded at: .../mas_project/data/ecommerce.db
```

---

## Running the System

### Run the full demo (5 test cases)

```bash
python main.py
```

You will see each customer message processed through all 4 agents with the final response printed.

---

## Running the Tests

### All tests (unit + integration + LLM-as-a-Judge)

```bash
pytest tests/test_mas.py -v
```

### Only unit tests (fast, no LLM calls)

```bash
pytest tests/test_mas.py -v -k "not TestLLMJudge and not TestGraphIntegration"
```

### Only integration + judge tests

```bash
pytest tests/test_mas.py -v -k "TestGraphIntegration or TestLLMJudge"
```

Expected output (all passing):
```
tests/test_mas.py::TestOrderLookupTool::test_valid_order            PASSED
tests/test_mas.py::TestOrderLookupTool::test_invalid_order          PASSED
tests/test_mas.py::TestOrderLookupTool::test_sql_injection_resistance PASSED
tests/test_mas.py::TestInventoryLookupTool::test_in_stock_item      PASSED
...
tests/test_mas.py::TestLLMJudge::test_judge_refund_response_quality PASSED
tests/test_mas.py::TestLLMJudge::test_judge_no_internal_leakage     PASSED
tests/test_mas.py::TestLLMJudge::test_judge_general_query_no_order  PASSED
```

---

## Viewing the Trace Log

Every agent action is logged to `logs/agent_trace.jsonl`:

```bash
# macOS / Linux
cat logs/agent_trace.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(r['timestamp'], r['agent'], r['stage'])
"

# Windows PowerShell
Get-Content logs\agent_trace.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).agent + ' ' + ($_ | ConvertFrom-Json).stage }
```

---

## Agent Overview

| Agent | File | Responsibility |
|-------|------|----------------|
| Triage Agent | `src/agents.py` | Classifies intent, extracts order ID, routes workflow |
| Order Intelligence Agent | `src/agents.py` | Fetches order from SQLite |
| Policy Agent | `src/agents.py` | Applies business rules, decides approve/reject |
| Action & Response Agent | `src/agents.py` | Executes action, writes customer response, saves ticket |

## Tool Overview

| Tool | File | Used By |
|------|------|---------|
| `lookup_order_tool` | `src/tools.py` | Order Intelligence Agent |
| `lookup_inventory_tool` | `src/tools.py` | Action/Response Agent |
| `lookup_policy_tool` | `src/tools.py` | Policy Agent |
| `calculate_refund_tool` | `src/tools.py` | Action/Response Agent |
| `save_ticket_tool` | `src/tools.py` | Action/Response Agent |

---

## Workflow Diagram

```
Customer Message
       │
       ▼
 ┌─────────────┐
 │ Triage Agent│  ── classifies intent, extracts order ID
 └──────┬──────┘
        │
   Order ID found?
   ┌────┴────┐
  YES       NO
   │         └──────────────────────┐
   ▼                                │
 ┌─────────────────────────┐        │
 │ Order Intelligence Agent│        │
 └────────────┬────────────┘        │
              │                     │
              ▼                     │
       ┌─────────────┐              │
       │ Policy Agent│              │
       └──────┬──────┘              │
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │ Action & Response Agent│
              └───────────┬────────────┘
                          │
                          ▼
                        END
                  (ticket saved to DB)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama: command not found` | Restart terminal after Ollama install |
| `Connection refused` on port 11434 | Run `ollama serve` in a separate terminal |
| `ModuleNotFoundError: src` | Make sure you are running commands from inside `mas_project/` |
| Slow responses | Normal for 3B model on CPU — first call warms up the model (~20–30 sec) |
| `pydantic` version errors | Ensure `pip install pydantic>=2.0.0` |

---

## Python 3.12 Compatibility Note

✅ All dependencies are fully compatible with Python 3.12.
- `typing_extensions` is included as a safety net
- `dict[str, Any]` and `list[str]` built-in generics are used (Python 3.9+ syntax)
- `datetime.now(timezone.utc)` replaces deprecated `datetime.utcnow()`
- No deprecated APIs used
