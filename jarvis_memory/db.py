"""
Database connection and schema setup.

SQLite is a single file, jarvis.db, with no server process. That is the right
shape for this project: the board boots, the file is there, nothing to start
or supervise.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path

from .embeddings import DIM

log = logging.getLogger(__name__)

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
        # Migrate BEFORE applying the schema: schema.sql creates indexes on
        # columns that an older database does not have yet, and CREATE INDEX
        # fails on a missing column where CREATE TABLE IF NOT EXISTS would
        # have silently done nothing.
        _migrate(conn)
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _load_vec(conn)
        _local.conn = conn
    return conn


# Columns added after the first release. CREATE TABLE IF NOT EXISTS does
# nothing for a table that already exists, so an existing jarvis.db on the
# board would be missing them; ALTER TABLE ADD COLUMN fills them in. Every
# entry needs a DEFAULT (or be nullable) for ALTER to accept it.
_ADDED_COLUMNS = {
    "events": {
        "source": "TEXT NOT NULL DEFAULT 'seed'",
        "uid": "TEXT",
        "ends_at": "TEXT",
        "all_day": "INTEGER NOT NULL DEFAULT 0",
    },
    "facts": {
        "polarity": "TEXT",
        "subject": "TEXT",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not present:
            continue        # fresh database: schema.sql creates the table complete
        for column, ddl in columns.items():
            if column not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                log.info("migrated: %s.%s", table, column)
    conn.commit()


def _load_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec and create the vector table, if both are possible.

    The vector table cannot live in schema.sql: CREATE VIRTUAL TABLE ... USING
    vec0 only parses once the extension is loaded, so it is created here.

    Failure is not fatal. Some Python builds ship sqlite3 with extension
    loading compiled out, and a teammate may simply not have installed the
    package -- in both cases the module still runs, with fact lookup falling
    back to a non-semantic query.
    """
    # The flag lives on the thread-local, not the connection:
    # sqlite3.Connection has no __dict__ and rejects new attributes.
    _local.vec_enabled = False
    try:
        import sqlite_vec
    except ImportError:
        log.info("sqlite-vec not installed; semantic search disabled")
        return

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        log.info("could not load sqlite-vec (%s); semantic search disabled", exc)
        return

    # user_id as a partition key so a search is scoped to one person inside
    # the index itself. Filtering after the fact would be wrong: the nearest
    # k rows overall may contain none of this user's.
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
               fact_id INTEGER PRIMARY KEY,
               user_id TEXT partition key,
               embedding float[{DIM}]
           )"""
    )
    # created_at is a metadata column, not a partition key: sqlite-vec can
    # apply "created_at < ?" inside the nearest-neighbour search, so recent
    # turns (already in the live history) are excluded before k is counted
    # rather than filtered out afterwards, leaving fewer than k results.
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_dialogue USING vec0(
               dialogue_id INTEGER PRIMARY KEY,
               user_id TEXT partition key,
               created_at INTEGER,
               embedding float[{DIM}]
           )"""
    )
    _local.vec_enabled = True


def vec_available() -> bool:
    """True if vector search is usable on this thread's connection."""
    connect()
    return getattr(_local, "vec_enabled", False)


def is_seeded() -> bool:
    """True if there is at least one user in the database."""
    return connect().execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def close() -> None:
    """Close this thread's connection. Mainly useful in tests."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.vec_enabled = False
