import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    patient_label TEXT,
    created_at REAL NOT NULL,
    result_json TEXT NOT NULL,
    thumbnail TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def save_scan(filename: str, patient_label: str, result: dict, thumbnail_b64: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO scans (filename, patient_label, created_at, result_json, thumbnail) VALUES (?, ?, ?, ?, ?)",
            (filename, patient_label, time.time(), json.dumps(result), thumbnail_b64),
        )
        return cur.lastrowid


def list_scans(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, patient_label, created_at, result_json, thumbnail FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "filename": r["filename"],
                "patient_label": r["patient_label"],
                "created_at": r["created_at"],
                "result": json.loads(r["result_json"]),
                "thumbnail": r["thumbnail"],
            })
        return out


def get_scan(scan_id: int):
    with get_conn() as conn:
        r = conn.execute(
            "SELECT id, filename, patient_label, created_at, result_json, thumbnail FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "filename": r["filename"],
            "patient_label": r["patient_label"],
            "created_at": r["created_at"],
            "result": json.loads(r["result_json"]),
            "thumbnail": r["thumbnail"],
        }


def stats_summary():
    with get_conn() as conn:
        rows = conn.execute("SELECT result_json FROM scans").fetchall()
    total = len(rows)
    verdict_counts = {}
    for r in rows:
        v = json.loads(r["result_json"]).get("final_verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    return {"total_scans": total, "verdict_counts": verdict_counts}
