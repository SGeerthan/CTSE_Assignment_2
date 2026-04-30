"""
models.py — Pydantic input models for tools and the shared LangGraph MASState.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ── Tool input models ─────────────────────────────────────────────────────────

class AccountLookupInput(BaseModel):
    """Input model for the account lookup tool."""
    account_id: str = Field(..., description="The account ID to fetch, e.g. ACC001")


class TransactionLookupInput(BaseModel):
    """Input model for the transaction lookup tool."""
    transaction_id: str = Field(..., description="Transaction ID to fetch, e.g. TXN001")


class PolicyLookupInput(BaseModel):
    """Input model for the policy lookup tool."""
    policy_name: str = Field(
        ..., description="Policy name: dispute_policy | card_block_policy | account_access_policy"
    )


class DisputeAmountInput(BaseModel):
    """Input model for the disputed amount helper."""
    amount: float = Field(..., ge=0, description="Disputed transaction amount in LKR")


class SaveTicketInput(BaseModel):
    """Input model for saving a resolved support ticket."""
    ticket_id:        str
    customer_message: str
    final_decision:   str
    final_response:   str
    account_id:       Optional[str] = None


# ── Shared graph state ────────────────────────────────────────────────────────

class MASState(TypedDict, total=False):
    """
    Global state object passed between every LangGraph node.
    Fields are Optional so each node only sets what it owns.
    """
    ticket_id:          str
    customer_message:   str
    extracted_account_id: Optional[str]
    extracted_transaction_id: Optional[str]
    triage_label:       Optional[str]
    account_details:    Optional[dict[str, Any]]
    transaction_details: Optional[dict[str, Any]]
    policy_decision:    Optional[dict[str, Any]]
    action_result:      Optional[dict[str, Any]]
    final_response:     Optional[str]
    routing_reason:     Optional[str]
    errors:             list[str]
