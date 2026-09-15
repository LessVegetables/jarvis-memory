"""
Database connection and schema setup.

SQLite is a single file, jarvis.db, with no server process. That is the right
shape for this project: the board boots, the file is there, nothing to start
or supervise.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = _PACKAGE_DIR / "schema.sql"
DEFAULT_DB_PATH = _PACKAGE_DIR.parent / "jarvis.db"

# One connection per thread. sqlite3 connections cannot be shared across
# threads, and the orchestrator will almost certainly have more than one
# (audio capture tends to run on its own). Thread-local storage sidesteps
# the whole problem for about ten lines.
_local = threading.local()


def db_path() -> Path:
    """Where the database file lives. Override with the JARVIS_DB env var."""
    return Path(os.environ.get("JARVIS_DB", DEFAULT_DB_PATH))


def connect() -> sqlite3.Connection:
    """Return this thread's connection, creating the schema if needed."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(db_path())
        # Rows behave like dicts: row["name"] instead of row[0].
        conn.row_factory = sqlite3.Row
        # Foreign keys are OFF by default in SQLite and the setting is
        # per-connection, not stored in the file. Without this, ON DELETE
        # CASCADE silently does nothing.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _local.conn = conn
    return conn


def is_seeded() -> bool:
    """True if there is at least one user in the database."""
    return connect().execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def close() -> None:
    """Close this thread's connection. Mainly useful in tests."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
