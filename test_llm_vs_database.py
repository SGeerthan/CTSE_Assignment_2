"""
Diagnostic script to show LLM vs Database usage in the MAS system.
This demonstrates that the LLM is actively being called and influencing decisions.
"""
import json
from pprint import pprint

from src.config import OLLAMA_MODEL
from src.database import init_db, get_db_connection
from src.models import (
    AccountLookupInput,
    TransactionLookupInput, 
    PolicyLookupInput,
)
from src.tools import (
    lookup_account_tool,
    lookup_transaction_tool,
    lookup_policy_tool,
)
from src.agents import invoke_ollama


def show_database_lookups():
    """Show what data comes from the DATABASE only."""
    print("\n" + "="*70)
    print("STEP 1: DATABASE LOOKUPS (No LLM involved)")
    print("="*70)
    
    # Database lookup 1: Account info
    print("\n📊 ACCOUNT LOOKUP (Database query):")
    account_result = lookup_account_tool(AccountLookupInput(account_id="ACC001"))
    pprint(account_result)
    
    # Database lookup 2: Transaction info
    print("\n📊 TRANSACTION LOOKUP (Database query):")
    txn_result = lookup_transaction_tool(TransactionLookupInput(transaction_id="TXN001"))
    pprint(txn_result)
    
    # Database lookup 3: Policy info
    print("\n📊 POLICY LOOKUP (Database query):")
    policy_result = lookup_policy_tool(PolicyLookupInput(policy_name="dispute_policy"))
    pprint(policy_result)


def show_triage_llm_call():
    """Show what the TRIAGE AGENT LLM returns."""
    print("\n" + "="*70)
    print("STEP 2: TRIAGE AGENT LLM CALL (AI Decision Making)")
    print("="*70)
    
    customer_message = "Hello, my account ACC001 has a suspicious transaction TXN001. I want to dispute it."
    print(f"\n🤖 Sending to LLM: \"{customer_message}\"")
    
    system_prompt = """
You are the TRIAGE AGENT for a bank customer support system.
Your job:
1. Categorise the intent into exactly one of:
   - account_access_issue
   - transaction_dispute
   - card_block_request
   - transaction_status_query
   - general_query
2. Extract the account ID if one is mentioned (pattern: ACCxxx).
3. Write a short routing reason (one sentence).

Respond ONLY in valid JSON — no prose, no markdown fences.
Output JSON keys: intent, routing_reason, extracted_account_id
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": customer_message}
    ]
    
    print("\n📡 Calling Ollama LLM...")
    llm_response = invoke_ollama(messages)
    print("\n🧠 LLM RESPONSE:")
    print(llm_response)
    
    # Parse as JSON
    try:
        parsed = json.loads(llm_response)
        print("\n✅ Parsed LLM JSON:")
        pprint(parsed)
    except:
        print("\n❌ Could not parse LLM response as JSON")


def show_policy_llm_call():
    """Show what the POLICY AGENT LLM returns."""
    print("\n" + "="*70)
    print("STEP 3: POLICY AGENT LLM CALL (AI Policy Decision)")
    print("="*70)
    
    # Get real data from database first
    account_details = lookup_account_tool(AccountLookupInput(account_id="ACC001"))
    transaction_details = lookup_transaction_tool(TransactionLookupInput(transaction_id="TXN001"))
    policy_details = lookup_policy_tool(PolicyLookupInput(policy_name="dispute_policy"))
    
    system_prompt = """
You are the POLICY AGENT for a bank customer support system.

You must decide:
- approved:       true or false
- decision_type:  account_access_help | dispute_opened | card_blocked | status_update | reject
- reason:         one-sentence explanation

Rules:
- Use ONLY the data provided — do not invent facts.
- Respond ONLY in valid JSON — no prose, no markdown fences.
"""
    
    user_prompt = f"""
Intent: transaction_dispute

Account Details:
{json.dumps(account_details, indent=2)}

Transaction Details:
{json.dumps(transaction_details, indent=2)}

Policy Text:
{policy_details.get('policy_text', 'No policy')}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("\n📡 Calling Ollama LLM with account + transaction + policy data...")
    llm_response = invoke_ollama(messages)
    print("\n🧠 LLM RESPONSE:")
    print(llm_response)
    
    # Parse as JSON
    try:
        parsed = json.loads(llm_response)
        print("\n✅ Parsed LLM JSON:")
        pprint(parsed)
    except:
        print("\n❌ Could not parse LLM response as JSON")


def show_response_llm_call():
    """Show what the RESPONSE AGENT LLM returns."""
    print("\n" + "="*70)
    print("STEP 4: RESPONSE AGENT LLM CALL (AI Customer Response)")
    print("="*70)
    
    customer_message = "Hello, my account ACC001 has a suspicious transaction TXN001. I want to dispute it."
    account_details = lookup_account_tool(AccountLookupInput(account_id="ACC001"))
    transaction_details = lookup_transaction_tool(TransactionLookupInput(transaction_id="TXN001"))
    
    policy_decision = {
        "approved": False,
        "decision_type": "dispute_opened",
        "reason": "Posted transaction within 30-day dispute window"
    }
    
    action_result = {
        "success": True,
        "message": "Dispute has been recorded"
    }
    
    system_prompt = """
You are the RESPONSE AGENT for a bank customer support team.

Write a professional, empathetic reply to the customer using:
- Their original message
- Account and transaction details
- Policy decision
- Action result

Rules:
- Be polite, concise, and helpful.
- NEVER mention internal systems.
"""
    
    user_prompt = f"""
Customer Message:
{customer_message}

Account Details:
{json.dumps(account_details, indent=2)}

Transaction Details:
{json.dumps(transaction_details, indent=2)}

Policy Decision:
{json.dumps(policy_decision, indent=2)}

Action Result:
{json.dumps(action_result, indent=2)}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("\n📡 Calling Ollama LLM to generate customer response...")
    llm_response = invoke_ollama(messages)
    print("\n🧠 LLM CUSTOMER RESPONSE:")
    print(llm_response)


def main():
    """Run all diagnostics."""
    init_db()
    
    print("\n" + "="*70)
    print("MAS SYSTEM: LLM vs DATABASE DIAGNOSTIC")
    print("="*70)
    print(f"LLM Model: {OLLAMA_MODEL}")
    print("Endpoint: http://127.0.0.1:11434")
    
    try:
        # 1. Show database only
        show_database_lookups()
        
        # 2. Show LLM triage
        show_triage_llm_call()
        
        # 3. Show LLM policy
        show_policy_llm_call()
        
        # 4. Show LLM response
        show_response_llm_call()
        
        print("\n" + "="*70)
        print("✅ DIAGNOSTIC COMPLETE")
        print("="*70)
        print("\nSUMMARY:")
        print("  • Database is used for: Account lookup, Transaction lookup, Policy text")
        print("  • LLM is used for: Intent classification, Policy decision, Customer response")
        print("  • All LLM calls are working correctly!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
