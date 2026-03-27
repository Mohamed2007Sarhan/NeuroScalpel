"""
session_manager.py
==================
Manages per-pipeline-run SQLite session databases stored in a dedicated
sessions/ folder. Each time the user clicks START WORD a new session is
created with its own isolated folder and database.

Schema
------
sessions   — top-level record for the run (model, order, timestamp)
scan_results — one row per Phase-2 scan (layer deviations, critical layer)
targets      — Phase-3 locked targets (layer, vector point)
edits        — Phase-5 ROME/LyapLock edits applied (before/after checksums)
tasks        — individual tasks parsed from the user order (multi-task queue)
"""

import os
import uuid
import sqlite3
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("NeuroScalpel.SessionManager")

# Root folder that holds all session subfolders (next to main.py)
SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"


class SessionManager:
    """
    Singleton-style session manager.
    Call create_session() at the start of each pipeline run.
    All subsequent log_* calls persist data for that session.
    """

    def __init__(self):
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, model_name: str, order_text: str) -> str:
        """
        Creates a new session folder + SQLite DB.
        Returns the session_id (UUID).
        """
        # Close any existing session first
        self.close()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = f"{ts}_{uuid.uuid4().hex[:8]}"
        self._session_dir = SESSIONS_ROOT / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._session_dir / "session.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # safe for multi-thread reads

        self._create_schema()
        self._insert_session(model_name, order_text)

        logger.info(f"Session created: {self._session_id} -> {self._db_path}")
        return self._session_id

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def db_path(self) -> Optional[Path]:
        return self._db_path

    def log_task(self, task_index: int, trick_prompt: str, subject: str,
                 wrong_value: str, correct_value: str) -> int:
        """Inserts a task record. Returns the new row id."""
        self._require_session()
        cur = self._conn.execute(
            """INSERT INTO tasks
               (session_id, task_index, trick_prompt, subject, wrong_value, correct_value, status)
               VALUES (?,?,?,?,?,?,?)""",
            (self._session_id, task_index, trick_prompt, subject,
             wrong_value, correct_value, "queued")
        )
        self._conn.commit()
        return cur.lastrowid

    def update_task_status(self, task_index: int, status: str):
        """Updates task status: queued | running | done | failed"""
        self._require_session()
        self._conn.execute(
            "UPDATE tasks SET status=? WHERE session_id=? AND task_index=?",
            (status, self._session_id, task_index)
        )
        self._conn.commit()

    def log_scan_result(self, task_index: int, critical_layer: str,
                        max_deviation: float, raw_report: str):
        """Saves Phase-2 scan results."""
        self._require_session()
        self._conn.execute(
            """INSERT INTO scan_results
               (session_id, task_index, critical_layer, max_deviation, raw_report)
               VALUES (?,?,?,?,?)""",
            (self._session_id, task_index, critical_layer,
             max_deviation, raw_report)
        )
        self._conn.commit()

    def log_target(self, task_index: int, layer_idx: int, vector_point: int,
                   analysis_summary: str):
        """Saves Phase-3 target lock."""
        self._require_session()
        self._conn.execute(
            """INSERT INTO targets
               (session_id, task_index, layer_idx, vector_point, analysis_summary)
               VALUES (?,?,?,?,?)""",
            (self._session_id, task_index, layer_idx, vector_point,
             analysis_summary)
        )
        self._conn.commit()

    def log_edit(self, task_index: int, method: str,
                 weights_changed: str, success: bool, notes: str = ""):
        """Saves a ROME/LyapLock edit record."""
        self._require_session()
        self._conn.execute(
            """INSERT INTO edits
               (session_id, task_index, method, weights_changed, success, notes)
               VALUES (?,?,?,?,?,?)""",
            (self._session_id, task_index, method,
             weights_changed, int(success), notes)
        )
        self._conn.commit()

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns a dict summary of the current session for display."""
        self._require_session()
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id=?",
            (self._session_id,)
        ).fetchone()
        tasks = self._conn.execute(
            "SELECT * FROM tasks WHERE session_id=? ORDER BY task_index",
            (self._session_id,)
        ).fetchall()
        return {
            "session": dict(row) if row else {},
            "tasks": [dict(t) for t in tasks]
        }

    def close(self):
        """Closes the current DB connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_session(self):
        if not self._conn or not self._session_id:
            raise RuntimeError("No active session. Call create_session() first.")

    def _create_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                model_name   TEXT,
                order_text   TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT,
                task_index    INTEGER,
                trick_prompt  TEXT,
                subject       TEXT,
                wrong_value   TEXT,
                correct_value TEXT,
                status        TEXT DEFAULT 'queued',
                created_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scan_results (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT,
                task_index     INTEGER,
                critical_layer TEXT,
                max_deviation  REAL,
                raw_report     TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS targets (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT,
                task_index       INTEGER,
                layer_idx        INTEGER,
                vector_point     INTEGER,
                analysis_summary TEXT,
                created_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS edits (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                task_index      INTEGER,
                method          TEXT,
                weights_changed TEXT,
                success         INTEGER,
                notes           TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    def _insert_session(self, model_name: str, order_text: str):
        self._conn.execute(
            "INSERT INTO sessions (session_id, model_name, order_text) VALUES (?,?,?)",
            (self._session_id, model_name, order_text)
        )
        self._conn.commit()


# Global singleton instance used across the app
session_manager = SessionManager()
