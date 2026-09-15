"""
Access to user data, backed by SQLite.

The function signatures here are the contract the rest of the module is
written against. They did not change when the hardcoded dicts were replaced
by real queries in step 2, and they will not change again when fact lookup
becomes vector search in step 5. Only the bodies move.

Run `python3 -m jarvis_memory.seed` to fill the database with dev data.
"""

from __future__ import annotations

from datetime import date

from . import db


def get_profile(user_id: str) -> dict | None:
    """Return the user's profile, or None if no such user_id exists."""
    row = db.connect().execute(
        "SELECT user_id, name, age FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def get_schedule(user_id: str, day: date) -> list[tuple[str, str]]:
    """Events for the given day: [(time, title), ...], earliest first.

    This is a lookup by key, not a search by meaning. A schedule has rigid
    structure, so RAG would be not just unnecessary here but harmful: this
    query either returns the right day or nothing. It cannot "almost guess"
    and hand back Wednesday.

    date(starts_at) strips the time off the ISO string so the comparison is
    against a whole day.
    """
    rows = db.connect().execute(
        "SELECT starts_at, title, location FROM events "
        "WHERE user_id = ? AND date(starts_at) = ? "
        "ORDER BY starts_at",
        (user_id, day.isoformat()),
    ).fetchall()
    return [(row["starts_at"][11:16], _describe(row)) for row in rows]


def get_facts(user_id: str, transcript: str, limit: int = 3) -> list[str]:
    """Facts about the user that are relevant to their question.

    FOR NOW: returns the `limit` most recent facts and ignores the question.
    STEP 5: vector search goes here (sqlite-vec over embeddings of
    facts.text), and `transcript` finally starts to matter.

    The default limit is 3 on purpose. There are eight facts per person in
    the database, and all eight would fit in the context window -- but a
    1.5B model handles three relevant facts better than eight mostly
    irrelevant ones. Picking the right three is exactly what step 5 buys.
    """
    rows = db.connect().execute(
        "SELECT text FROM facts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [row["text"] for row in rows]


def _describe(row) -> str:
    """'Лекция по матанализу, ауд. 305' -- title plus location when there is one."""
    return f"{row['title']}, {row['location']}" if row["location"] else row["title"]
