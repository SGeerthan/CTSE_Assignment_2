"""
agents.py — The four MAS agents:
  1. triage_agent               — classifies intent and extracts account ID
  2. account_intelligence_agent — fetches account data from SQLite
  3. policy_agent             — applies business rules
  4. action_response_agent    — executes action + writes final response
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain_core.prompts import ChatPromptTemplate

from src.config import OLLAMA_MODEL, OLLAMA_TEMPERATURE
from src.logger import log_event
from src.models import (
    TransactionLookupInput,
    MASState,
    AccountLookupInput,
    PolicyLookupInput,
    DisputeAmountInput,
    SaveTicketInput,
)
from src.tools import (
    calculate_dispute_credit_tool,
    lookup_transaction_tool,
    lookup_account_tool,
    lookup_policy_tool,
    save_ticket_tool,
)

LLM_TIMEOUT_SECONDS = 90


def invoke_ollama(messages: list[dict[str, str]], timeout_s: int = LLM_TIMEOUT_SECONDS) -> str:
    """
    Call local Ollama chat API with a strict timeout.
    """
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": OLLAMA_TEMPERATURE},
        }
    ).encode("utf-8")
    req = Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama call timed out/failed after {timeout_s}s: {exc}") from exc

    content = (((parsed or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content


def invoke_prompt_with_timeout(
    prompt_template: ChatPromptTemplate,
    payload: dict[str, Any],
    timeout_s: int = LLM_TIMEOUT_SECONDS,
) -> str:
    """
    Render prompt template and send to Ollama with timeout.
    """
    rendered = prompt_template.format_messages(**payload)
    messages = [{"role": msg.type, "content": msg.content} for msg in rendered]
    # LangChain uses "human"; Ollama expects "user".
    for message in messages:
        if message["role"] == "human":
            message["role"] = "user"
    return invoke_ollama(messages, timeout_s=timeout_s)


def invoke_llm_with_timeout(
    prompt_template: ChatPromptTemplate,
    payload: dict[str, Any],
    timeout_s: int = LLM_TIMEOUT_SECONDS,
):
    """
    Backwards-compatible wrapper that returns a string response.
    """
    return invoke_prompt_with_timeout(prompt_template, payload, timeout_s=timeout_s)


# ── Utilities ─────────────────────────────────────────────────────────────────

def extract_account_id(text: str) -> Optional[str]:
    """
    Extract an account ID (pattern ACCxxx) from free-form text.

    Args:
        text: Raw customer message.

    Returns:
        Matched account ID string, e.g. 'ACC001', or None.
    """
    match = re.search(r"\bACC\d{3}\b", text.upper())
    return match.group(0) if match else None


def extract_transaction_id(text: str) -> Optional[str]:
    """
    Extract a transaction ID (pattern TXNxxx) from free-form text.
    """
    match = re.search(r"\bTXN\d{3}\b", text.upper())
    return match.group(0) if match else None


def safe_json_loads(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output, stripping markdown fences if present.

    Args:
        text: Raw LLM output expected to be a JSON object.

    Returns:
        Parsed dict, or {'parse_error': True, 'raw': text} on failure.
    """
    # Strip ```json ... ``` fences that some models emit
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"parse_error": True, "raw": text}


# ── Agent 1: Triage ───────────────────────────────────────────────────────────

_triage_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the TRIAGE AGENT for a bank customer support system.

Your job:
1. Read the customer message carefully.
2. Categorise the intent into exactly one of:
   - account_access_issue
   - transaction_dispute
   - card_block_request
   - transaction_status_query
   - general_query
3. Extract the account ID if one is mentioned (pattern: ACCxxx).
4. Write a short routing reason (one sentence).

Rules:
- Respond ONLY in valid JSON — no prose, no markdown fences.
- Never invent an account ID.

Output JSON keys: intent, routing_reason, extracted_account_id
"""),
    ("human", "{customer_message}"),
])


def triage_agent(state: MASState) -> MASState:
    """
    Classify the customer's intent and extract the account ID.

    Populates: triage_label, routing_reason, extracted_account_id
    """
    log_event("triage_agent", "start", {"message": state["customer_message"]})

    # Regex fallback so account ID is captured even if LLM misses it
    message = state["customer_message"]
    regex_account_id = extract_account_id(message)
    regex_transaction_id = extract_transaction_id(message)

    try:
        content = invoke_llm_with_timeout(_triage_prompt, {"customer_message": state["customer_message"]})
        parsed = safe_json_loads(content)
    except Exception as exc:
        parsed = {"intent": "general_query", "routing_reason": str(exc), "extracted_account_id": regex_account_id}
        errors: list[str] = list(state.get("errors") or [])
        errors.append(f"Triage LLM failed: {exc}")
        state["errors"] = errors

    triage_label      = parsed.get("intent", "general_query")
    routing_reason    = parsed.get("routing_reason", "No reason generated.")
    extracted_account_id = parsed.get("extracted_account_id") or regex_account_id

    log_event("triage_agent", "end", {
        "triage_label":       triage_label,
        "routing_reason":     routing_reason,
        "extracted_account_id": extracted_account_id,
        "extracted_transaction_id": regex_transaction_id,
    })

    state["triage_label"]       = triage_label
    state["routing_reason"]     = routing_reason
    state["extracted_account_id"] = extracted_account_id
    state["extracted_transaction_id"] = regex_transaction_id
    return state


# ── Agent 2: Account Intelligence ─────────────────────────────────────────────

def account_intelligence_agent(state: MASState) -> MASState:
    """
    Look up the account referenced in the customer message.

    Uses: lookup_account_tool
    Populates: account_details, errors (on failure)
    """
    log_event("account_intelligence_agent", "start",
              {"extracted_account_id": state.get("extracted_account_id")})

    account_id = state.get("extracted_account_id")
    errors: list[str] = list(state.get("errors") or [])

    if not account_id:
        err_msg = "No account ID found in customer message."
        errors.append(err_msg)
        state["account_details"] = {"found": False, "error": err_msg}
        state["errors"]        = errors
        log_event("account_intelligence_agent", "end", state["account_details"])
        return state

    account_result = lookup_account_tool(AccountLookupInput(account_id=account_id))
    state["account_details"] = account_result

    if not account_result.get("found"):
        err_msg = account_result.get("error", "Account lookup returned no data.")
        errors.append(err_msg)
        state["errors"] = errors

    txn_id = state.get("extracted_transaction_id")
    if txn_id:
        txn_result = lookup_transaction_tool(TransactionLookupInput(transaction_id=txn_id))
        if txn_result.get("found") and txn_result.get("account_id") == account_id:
            state["transaction_details"] = txn_result
        elif txn_result.get("found"):
            state["transaction_details"] = {"found": False, "error": "Transaction does not belong to this account."}
            errors.append("Transaction/account mismatch.")
            state["errors"] = errors
        else:
            state["transaction_details"] = txn_result
            errors.append(txn_result.get("error", "Transaction lookup returned no data."))
            state["errors"] = errors
    else:
        state["transaction_details"] = {"found": False, "error": "No transaction ID found in message."}

    log_event("account_intelligence_agent", "tool_call_lookup_account", account_result)
    log_event("account_intelligence_agent", "tool_call_lookup_transaction", state.get("transaction_details"))
    return state


# ── Agent 3: Policy ───────────────────────────────────────────────────────────

_policy_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the POLICY AGENT for a bank customer support system.

You receive:
- The customer's support intent
- Verified account details
- The relevant policy text

You must decide:
- approved:       true or false
- decision_type:  account_access_help | dispute_opened | card_blocked | status_update | reject
- reason:         one-sentence explanation
- required_policy: the policy name used

Rules:
- Use ONLY the data provided — do not invent facts.
- Respond ONLY in valid JSON — no prose, no markdown fences.
"""),
    ("human", """
Intent:
{intent}

Account Details:
{account_details}

Transaction Details:
{transaction_details}

Policy Text:
{policy_text}
"""),
])


def policy_agent(state: MASState) -> MASState:
    """
    Apply the relevant business policy and decide whether to approve the request.

    Uses: lookup_policy_tool
    Populates: policy_decision
    """
    log_event("policy_agent", "start", {
        "triage_label":  state.get("triage_label"),
        "account_details": state.get("account_details"),
    })

    intent        = state.get("triage_label", "general_query")
    account_details = state.get("account_details") or {}
    transaction_details = state.get("transaction_details") or {}

    # If account was not verified, reject immediately
    if not account_details.get("found"):
        state["policy_decision"] = {
            "approved":       False,
            "decision_type":  "reject",
            "reason":         "Account could not be verified.",
            "required_policy": None,
        }
        log_event("policy_agent", "end", state["policy_decision"])
        return state

    if intent in {"transaction_dispute", "transaction_status_query"} and not transaction_details.get("found"):
        state["policy_decision"] = {
            "approved": False,
            "decision_type": "reject",
            "reason": "Transaction could not be verified.",
            "required_policy": "dispute_policy",
        }
        log_event("policy_agent", "end", state["policy_decision"])
        return state

    # Choose the relevant policy
    policy_map = {
        "transaction_dispute":  "dispute_policy",
        "card_block_request":   "card_block_policy",
    }
    policy_name = policy_map.get(intent, "account_access_policy")

    policy_result = lookup_policy_tool(PolicyLookupInput(policy_name=policy_name))
    policy_text   = policy_result.get("policy_text", "")

    try:
        content = invoke_llm_with_timeout(
            _policy_prompt,
            {
                "intent": intent,
                "account_details": json.dumps(account_details, indent=2),
                "transaction_details": json.dumps(transaction_details, indent=2),
                "policy_text": policy_text,
            },
        )
        parsed = safe_json_loads(content)
    except Exception as exc:
        parsed = {
            "approved": False,
            "decision_type": "reject",
            "reason": f"Policy LLM failed: {exc}",
            "required_policy": policy_name,
        }
        errors: list[str] = list(state.get("errors") or [])
        errors.append(f"Policy LLM failed: {exc}")
        state["errors"] = errors

    if parsed.get("parse_error"):
        parsed = {
            "approved":       False,
            "decision_type":  "reject",
            "reason":         "Policy parsing failed.",
            "required_policy": policy_name,
        }

    parsed["required_policy"] = policy_name
    state["policy_decision"]  = parsed

    log_event("policy_agent", "tool_call_lookup_policy", policy_result)
    log_event("policy_agent", "end", parsed)
    return state


# ── Agent 4: Action + Response ────────────────────────────────────────────────

_response_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the RESPONSE AGENT for a bank customer support team.

Write a professional, empathetic reply to the customer using:
- Their original message
- The verified account and transaction details (if any)
- The policy decision
- The action result (dispute amount, account status, card status, etc.)

Rules:
- Be polite, concise, and helpful.
- NEVER mention internal systems: no "database", "agent", "model", "LangGraph", "Ollama", "SQLite".
- If the request is rejected, explain clearly and suggest next steps.
"""),
    ("human", """
Customer Message:
{customer_message}

Account Details:
{account_details}

Transaction Details:
{transaction_details}

Policy Decision:
{policy_decision}

Action Result:
{action_result}
"""),
])


def action_response_agent(state: MASState) -> MASState:
    """
    Execute the approved action (dispute handling / card status / account guidance) and
    compose the final customer-facing response.

    Uses: calculate_dispute_credit_tool, lookup_transaction_tool, save_ticket_tool
    Populates: action_result, final_response
    """
    log_event("action_response_agent", "start",
              {"policy_decision": state.get("policy_decision")})

    decision      = state.get("policy_decision") or {}
    account_details = state.get("account_details") or {}
    transaction_details = state.get("transaction_details") or {}
    action_result: dict[str, Any] = {}

    # ── Execute the approved action ───────────────────────────────────────────
    if decision.get("approved") and decision.get("decision_type") == "dispute_opened":
        if transaction_details.get("found"):
            action_result = calculate_dispute_credit_tool(
                DisputeAmountInput(amount=float(transaction_details["amount"]))
            )

    elif decision.get("decision_type") == "status_update":
        action_result = {
            "success": True,
            "message": f"Current transaction status: {transaction_details.get('status', 'Unknown')}",
        }
    elif decision.get("approved") and decision.get("decision_type") == "card_blocked":
        action_result = {"success": True, "message": "Your card has been blocked successfully."}
    elif decision.get("approved") and decision.get("decision_type") == "account_access_help":
        action_result = {
            "success": True,
            "message": f"Your account status is {account_details.get('account_status', 'Unknown')}. Follow verification prompts to restore access.",
        }

    else:
        action_result = {"success": True, "message": "No further action required."}

    state["action_result"] = action_result

    # ── Generate the customer-facing response ─────────────────────────────────
    try:
        final_response = invoke_llm_with_timeout(
            _response_prompt,
            {
                "customer_message": state["customer_message"],
                "account_details": json.dumps(account_details, indent=2),
                "transaction_details": json.dumps(transaction_details, indent=2),
                "policy_decision": json.dumps(decision, indent=2),
                "action_result": json.dumps(action_result, indent=2),
            },
        )
    except Exception as exc:
        final_response = (
            "We are currently experiencing a delay generating a full response. "
            "Please try again shortly. "
            f"(technical note: {exc})"
        )
        errors: list[str] = list(state.get("errors") or [])
        errors.append(f"Response LLM failed: {exc}")
        state["errors"] = errors

    state["final_response"] = final_response

    # ── Persist ticket ────────────────────────────────────────────────────────
    save_result = save_ticket_tool(
        SaveTicketInput(
            ticket_id=state["ticket_id"],
            customer_message=state["customer_message"],
            account_id=state.get("extracted_account_id"),
            final_decision=json.dumps(decision),
            final_response=final_response,
        )
    )

    log_event("action_response_agent", "tool_action_result",   action_result)
    log_event("action_response_agent", "tool_save_ticket",     save_result)
    log_event("action_response_agent", "end", {"final_response": final_response})

    return state
