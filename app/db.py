"""Penyimpanan riwayat prediksi (SQLite). Path diatur lewat env DB_PATH."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def db_path():
    return Path(os.getenv("DB_PATH", Path(__file__).resolve().parent.parent / "data" / "predictions.db"))


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            top1_disease TEXT NOT NULL,
            top1_probability REAL NOT NULL,
            top_predictions TEXT NOT NULL)""")


def save_prediction(model, symptoms, result):
    top = result["top_predictions"]
    with connect() as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at, model, symptoms, top1_disease, top1_probability, top_predictions)"
            " VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), model, json.dumps(sorted(symptoms)),
             top[0]["disease"], top[0]["probability"], json.dumps(top)))
        return cur.lastrowid


def list_predictions(limit=50):
    with connect() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "symptoms": json.loads(r["symptoms"]), "top_predictions": json.loads(r["top_predictions"])}
            for r in rows]


def stats():
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_model = {r[0]: r[1] for r in c.execute("SELECT model, COUNT(*) FROM predictions GROUP BY model")}
        top = [dict(r) for r in c.execute(
            "SELECT top1_disease AS disease, COUNT(*) AS n FROM predictions GROUP BY top1_disease "
            "ORDER BY n DESC LIMIT 5")]
    return {"total_predictions": total, "by_model": by_model, "top_diagnoses": top}
