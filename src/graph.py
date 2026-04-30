"""
graph.py — Builds and compiles the LangGraph multi-agent workflow.

Routing logic:
  Triage → (account ID found?)
               YES → Account Intelligence → Policy → Action/Response → END
               NO  → Action/Response → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents import (
    action_response_agent,
    account_intelligence_agent,
    policy_agent,
    triage_agent,
)
from src.logger import log_event
from src.models import MASState


def route_after_triage(state: MASState) -> str:
    """
    Conditional router executed after the Triage Agent.

    If an account ID was found, route through the full pipeline.
    Otherwise short-circuit directly to the Action/Response Agent.

    Args:
        state: Current MASState after triage.

    Returns:
        Node name to execute next.
    """
    label    = state.get("triage_label", "general_query")
    account_id = state.get("extracted_account_id")

    if not account_id:
        log_event("router", "route_decision", {
            "intent": label,
            "route":  "action_response_agent",
            "reason": "No account ID — skipping account lookup",
        })
        return "action_response_agent"

    log_event("router", "route_decision", {
        "intent": label,
        "route":  "account_intelligence_agent",
        "reason": "Account ID found — full pipeline",
    })
    return "account_intelligence_agent"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Returns:
        Compiled runnable graph ready to invoke.
    """
    workflow = StateGraph(MASState)

    # Register nodes
    workflow.add_node("triage_agent",             triage_agent)
    workflow.add_node("account_intelligence_agent", account_intelligence_agent)
    workflow.add_node("policy_agent",             policy_agent)
    workflow.add_node("action_response_agent",    action_response_agent)

    # Entry point
    workflow.set_entry_point("triage_agent")

    # Conditional routing after triage
    workflow.add_conditional_edges(
        "triage_agent",
        route_after_triage,
        {
            "account_intelligence_agent": "account_intelligence_agent",
            "action_response_agent":    "action_response_agent",
        },
    )

    # Fixed edges for the full pipeline
    workflow.add_edge("account_intelligence_agent", "policy_agent")
    workflow.add_edge("policy_agent",             "action_response_agent")
    workflow.add_edge("action_response_agent",    END)

    return workflow.compile()


# Module-level compiled app — import this wherever you need to run the graph
app = build_graph()
