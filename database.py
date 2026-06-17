import sqlite3
from datetime import datetime

DB_PATH = "finsight.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            company_name TEXT,
            direction TEXT NOT NULL,
            price_at_prediction REAL,
            target_price REAL,
            confidence INTEGER,
            timeframe_days INTEGER,
            reasoning TEXT,
            created_at TEXT,
            resolved INTEGER DEFAULT 0,
            outcome TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(ticker, company_name, direction, price, target_price, confidence, timeframe_days, reasoning):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (ticker, company_name, direction, price_at_prediction, target_price, confidence, timeframe_days, reasoning, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            company_name=excluded.company_name,
            direction=excluded.direction,
            price_at_prediction=excluded.price_at_prediction,
            target_price=excluded.target_price,
            confidence=excluded.confidence,
            timeframe_days=excluded.timeframe_days,
            reasoning=excluded.reasoning,
            created_at=excluded.created_at,
            resolved=0,
            outcome=NULL
    """, (ticker, company_name, direction, price, target_price, confidence, timeframe_days, reasoning, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_predictions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows