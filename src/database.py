"""
database.py — Creates and seeds the local SQLite database.
Run once before starting the application:
    python src/database.py
"""
from __future__ import annotations

import sqlite3
from src.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection to the local project database.

    Returns:
        sqlite3.Connection: Active database connection.
    """
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """
    Create all tables and seed initial data.
    Safe to call multiple times — uses INSERT OR REPLACE.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Accounts ──────────────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_id           TEXT PRIMARY KEY,
        customer_name        TEXT NOT NULL,
        account_type         TEXT NOT NULL,
        account_status       TEXT NOT NULL,
        balance              REAL NOT NULL
    )
    """)

    # ── Transactions ──────────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id   TEXT PRIMARY KEY,
        account_id       TEXT NOT NULL,
        merchant         TEXT NOT NULL,
        amount           REAL NOT NULL,
        status           TEXT NOT NULL,
        days_since_posted INTEGER NOT NULL,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )
    """)

    # ── Policies ──────────────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS policies (
        policy_name  TEXT PRIMARY KEY,
        policy_text  TEXT NOT NULL
    )
    """)

    # ── Tickets (written by agents at runtime) ────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id        TEXT PRIMARY KEY,
        customer_message TEXT NOT NULL,
        account_id       TEXT,
        final_decision   TEXT,
        final_response   TEXT
    )
    """)

    # ── Seed data ─────────────────────────────────────────────────────────────
    accounts = [
        ("ACC001", "Nimal Perera", "Savings",  "Active",   125000.50),
        ("ACC002", "Kavindi Silva", "Current",  "Active",    45200.00),
        ("ACC003", "Ahan Fernando", "Savings",  "Restricted", 8300.75),
        ("ACC004", "Sajini Jayasuriya", "Current", "Active",  2500.00),
    ]

    transactions = [
        ("TXN001", "ACC001", "City Supermarket", 4500.00, "Posted", 2),
        ("TXN002", "ACC002", "Online Electronics", 12500.00, "Pending", 1),
        ("TXN003", "ACC003", "Fuel Station", 9800.00, "Posted", 18),
        ("TXN004", "ACC001", "Streaming Service", 3200.00, "Posted", 35),
    ]

    policies = [
        ("dispute_policy",
         "Disputes can be raised for posted card transactions within 30 days of posting. "
         "Pending transactions cannot be disputed until posted."),
        ("card_block_policy",
         "Cards can be blocked immediately upon customer request. Emergency block requests "
         "must be completed during the same interaction."),
        ("account_access_policy",
         "Account access issues require account verification before actions are approved."),
    ]

    cursor.executemany("INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)", accounts)
    cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?,?)", transactions)
    cursor.executemany("INSERT OR REPLACE INTO policies VALUES (?,?)", policies)

    conn.commit()
    conn.close()
    print(f"[database] Seeded at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
