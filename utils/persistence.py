import sqlite3
import json
import pandas as pd

DB_NAME = "spendwise_data.db"


def _init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT,
        label TEXT,
        data TEXT,
        meta TEXT
    )
    """)

    conn.commit()
    conn.close()


_init_db()


def save_session(df, meta=None, label="Session"):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    data_json = df.to_json(orient="records", date_format="iso")
    meta_json = json.dumps(meta or {})

    cur.execute(
        """
        INSERT INTO sessions(saved_at, label, data, meta)
        VALUES(datetime('now'), ?, ?, ?)
        """,
        (label, data_json, meta_json),
    )

    conn.commit()
    conn.close()


def list_sessions():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    SELECT id, saved_at, label
    FROM sessions
    ORDER BY id DESC
    """)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "saved_at": r[1],
            "label": r[2],
        }
        for r in rows
    ]


def load_session(session_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "SELECT data, meta FROM sessions WHERE id=?",
        (session_id,)
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        raise ValueError("Session not found")

    df = pd.read_json(row[0])

    try:
        meta = json.loads(row[1])
    except:
        meta = {}

    return df, meta


def delete_session(session_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM sessions WHERE id=?",
        (session_id,)
    )

    conn.commit()
    conn.close()