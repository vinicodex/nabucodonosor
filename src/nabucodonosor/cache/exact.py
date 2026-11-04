import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _hash(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exact_cache (
            hash      TEXT PRIMARY KEY,
            answer    TEXT NOT NULL,
            sources   TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get(question: str, db_path: str, ttl_hours: int) -> str | None:
    try:
        conn = _connect(db_path)
        row = conn.execute(
            "SELECT answer, created_at FROM exact_cache WHERE hash = ?", (_hash(question),)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        created = datetime.fromisoformat(row[1])
        age_hours = (datetime.now(UTC) - created).total_seconds() / 3600
        if age_hours > ttl_hours:
            return None

        return str(row[0])
    except Exception:
        return None


def set(question: str, answer: str, sources: list[str], db_path: str) -> None:
    try:
        conn = _connect(db_path)
        sql = (
            "INSERT OR REPLACE INTO exact_cache "
            "(hash, answer, sources, created_at) VALUES (?,?,?,?)"
        )
        params = (_hash(question), answer, json.dumps(sources), datetime.now(UTC).isoformat())
        conn.execute(sql, params)
        conn.commit()
        conn.close()
    except Exception:
        pass


def purge_expired(db_path: str, ttl_hours: int) -> int:
    try:
        conn = _connect(db_path)
        cutoff = datetime.now(UTC)
        rows = conn.execute("SELECT hash, created_at FROM exact_cache").fetchall()
        to_delete = [
            r[0]
            for r in rows
            if (cutoff - datetime.fromisoformat(r[1])).total_seconds() / 3600 > ttl_hours
        ]
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            conn.execute(f"DELETE FROM exact_cache WHERE hash IN ({placeholders})", to_delete)
            conn.commit()
        conn.close()
        return len(to_delete)
    except Exception:
        return 0
