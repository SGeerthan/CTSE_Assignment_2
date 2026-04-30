"""
tools.py — Custom Python tools used by the MAS agents.
Each tool:
  • accepts a validated Pydantic model as input
  • returns a plain dict so agents can inspect the result
  • uses parameterised SQL (no injection risk)
  • has strict type hints and descriptive docstrings
"""
from __future__ import annotations

import sqlite3
from typing import Any

from src.database import get_db_connection
from src.models import (
    AccountLookupInput,
    TransactionLookupInput,
    PolicyLookupInput,
    DisputeAmountInput,
    SaveTicketInput,
)


# ── Tool 1: Account lookup ────────────────────────────────────────────────────

def lookup_account_tool(input_data: AccountLookupInput) -> dict[str, Any]:
    """
    Fetch a single account by account_id from the local SQLite database.

    Uses a parameterised query to prevent SQL injection.

    Args:
        input_data: Validated AccountLookupInput containing the account_id.

    Returns:
        dict with full account details when found, or an error payload.

    Example:
        >>> result = lookup_account_tool(AccountLookupInput(account_id="ACC001"))
        >>> result["found"]
        True
        >>> result["account_status"]
        'Active'
    """
    try:
        conn: sqlite3.Connection = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, customer_name, account_type, account_status, balance
            FROM   accounts
            WHERE  account_id = ?
            """,
            (input_data.account_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"found": False, "error": f"Account {input_data.account_id} not found."}

        return {
            "found":               True,
            "account_id":          row[0],
            "customer_name":       row[1],
            "account_type":        row[2],
            "account_status":      row[3],
            "balance":             row[4],
        }
    except Exception as exc:
        return {"found": False, "error": f"Account lookup failed: {exc}"}


# ── Tool 2: Transaction lookup ────────────────────────────────────────────────

def lookup_transaction_tool(input_data: TransactionLookupInput) -> dict[str, Any]:
    """
    Check details for a given transaction from the local transactions table.

    Args:
        input_data: Validated TransactionLookupInput containing transaction_id.

    Returns:
        dict with transaction fields, or an error payload.

    Example:
        >>> result = lookup_transaction_tool(TransactionLookupInput(transaction_id="TXN001"))
        >>> result["found"]
        True
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT transaction_id, account_id, merchant, amount, status, days_since_posted
            FROM transactions
            WHERE transaction_id = ?
            """,
            (input_data.transaction_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"found": False, "error": f"Transaction '{input_data.transaction_id}' not found."}

        return {
            "found":            True,
            "transaction_id":   row[0],
            "account_id":       row[1],
            "merchant":         row[2],
            "amount":           row[3],
            "status":           row[4],
            "days_since_posted": row[5],
        }
    except Exception as exc:
        return {"found": False, "error": f"Transaction lookup failed: {exc}"}


# ── Tool 3: Policy lookup ─────────────────────────────────────────────────────

def lookup_policy_tool(input_data: PolicyLookupInput) -> dict[str, Any]:
    """
    Fetch policy text from the local policies table by policy_name.

    Args:
        input_data: Validated PolicyLookupInput containing policy_name.

    Returns:
        dict with policy_name and policy_text when found, or an error payload.

    Example:
        >>> result = lookup_policy_tool(PolicyLookupInput(policy_name="refund_policy"))
        >>> "30 days" in result["policy_text"]
        True
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT policy_name, policy_text FROM policies WHERE policy_name = ?",
            (input_data.policy_name,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"found": False, "error": f"Policy '{input_data.policy_name}' not found."}

        return {
            "found":       True,
            "policy_name": row[0],
            "policy_text": row[1],
        }
    except Exception as exc:
        return {"found": False, "error": f"Policy lookup failed: {exc}"}


# ── Tool 4: Dispute amount helper ─────────────────────────────────────────────

def calculate_dispute_credit_tool(input_data: DisputeAmountInput) -> dict[str, Any]:
    """
    Calculate provisional dispute credit amount.

    Args:
        input_data: Validated DisputeAmountInput with transaction amount.

    Returns:
        dict with 'provisional_credit' in LKR and a 'success' flag.

    Example:
        >>> result = calculate_dispute_credit_tool(DisputeAmountInput(amount=4500.0))
        >>> result["provisional_credit"]
        4500.0
    """
    try:
        amount = round(input_data.amount, 2)
        return {"success": True, "provisional_credit": amount, "currency": "LKR"}
    except Exception as exc:
        return {"success": False, "error": f"Dispute credit calculation failed: {exc}"}


# ── Tool 5: Save ticket ───────────────────────────────────────────────────────

def save_ticket_tool(input_data: SaveTicketInput) -> dict[str, Any]:
    """
    Persist the final support ticket decision and response to SQLite.

    Called by the Action/Response Agent after composing the final reply.

    Args:
        input_data: Validated SaveTicketInput payload.

    Returns:
        dict with 'success' flag and the saved ticket_id.

    Example:
        >>> save_ticket_tool(SaveTicketInput(
        ...     ticket_id="T001", customer_message="...",
        ...     final_decision="{}", final_response="Done."
        ... ))
        {'success': True, 'ticket_id': 'T001'}
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO tickets
                (ticket_id, customer_message, account_id, final_decision, final_response)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                input_data.ticket_id,
                input_data.customer_message,
                input_data.account_id,
                input_data.final_decision,
                input_data.final_response,
            ),
        )
        conn.commit()
        conn.close()
        return {"success": True, "ticket_id": input_data.ticket_id}
    except Exception as exc:
        return {"success": False, "error": f"Ticket save failed: {exc}"}
