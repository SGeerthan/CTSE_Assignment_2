"""
app.py — Streamlit Chat UI for Multi-Agent Bank Support Resolver
Run: streamlit run app.py
"""

import streamlit as st
import uuid

from src.database import init_db
from src.models import MASState
from src.graph import app as graph_app
from main import check_ollama_ready

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Bank Support Resolver",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------------
# TITLE
# ----------------------------------
st.title("🏦 Multi-Agent Bank Support Resolver")
st.caption("AI-powered customer support using Multi-Agent System")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Interactive Chat", "Demo Cases"]
)

show_debug = st.sidebar.checkbox("Show Debug JSON")

if st.sidebar.button("Clear Chat"):
    st.session_state.history = []

# ----------------------------------
# CHECK OLLAMA
# ----------------------------------
ready, reason = check_ollama_ready()
if not ready:
    st.error("❌ Ollama not ready")
    st.code(reason)
    st.info("👉 Run: ollama serve")
    st.info("👉 Then: ollama pull <your-model>")
    st.stop()

# ----------------------------------
# INIT DB
# ----------------------------------
init_db()

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------
# DEMO DATA
# ----------------------------------
DEMO_MESSAGES = [
    "My account ACC001 has a suspicious transaction TXN001. I want to dispute it.",
    "Block my card for account ACC002 immediately.",
    "Status of transaction TXN002 for account ACC002?",
    "I cannot access my account ACC003, help me recover it.",
]

# ----------------------------------
# PROCESS FUNCTION
# ----------------------------------
def process_message(message):
    state: MASState = {
        "ticket_id": str(uuid.uuid4()),
        "customer_message": message,
        "errors": [],
    }

    with st.spinner("🤖 Processing your request..."):
        result = graph_app.invoke(state)

    st.session_state.history.append((message, result))


# ----------------------------------
# DEMO MODE
# ----------------------------------
if mode == "Demo Cases":
    st.subheader("📊 Demo Cases")

    for msg in DEMO_MESSAGES:
        if st.button(f"Run: {msg}"):
            process_message(msg)

# ----------------------------------
# CHAT INPUT
# ----------------------------------
if mode == "Interactive Chat":
    user_input = st.chat_input("Type your bank support message...")

    if user_input:
        process_message(user_input)

# ----------------------------------
# DISPLAY CHAT HISTORY
# ----------------------------------
for msg, result in st.session_state.history[::-1]:

    # USER MESSAGE
    with st.chat_message("user"):
        st.write(msg)

    # AI RESPONSE
    with st.chat_message("assistant"):
        st.success(result.get("final_response"))

        st.markdown("**🔍 Details:**")
        st.write(f"**Triage:** {result.get('triage_label')}")
        st.write(f"**Account ID:** {result.get('extracted_account_id')}")

        decision = result.get("policy_decision", {})
        st.write(f"**Decision:** {decision.get('decision_type')}")
        st.write(f"**Approved:** {decision.get('approved')}")

        st.write(f"**Action:** {result.get('action_result')}")

        if result.get("errors"):
            st.error(result["errors"])

        if show_debug:
            st.json(result)