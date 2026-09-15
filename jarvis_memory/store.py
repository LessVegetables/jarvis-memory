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

from . import db, embeddings


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

    Searches by meaning: the question is embedded and compared against the
    stored fact vectors, so "можно мне печенье с миндалём?" retrieves
    "аллергия на орехи" despite sharing no words with it.

    The default limit is 3 on purpose. All eight of a person's facts would
    fit in the context window -- but a 1.5B model answers better given three
    relevant facts than eight mostly irrelevant ones. Choosing the right
    three is the entire value of doing this.

    Falls back to the most recent facts when semantic search is unavailable
    (no model file, no sqlite-vec). The assistant still answers; it just
    stops being clever about which facts it mentions.
    """
    found = _search_facts(user_id, transcript, limit)
    if found is not None:
        return found

    rows = db.connect().execute(
        "SELECT text FROM facts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [row["text"] for row in rows]


def _search_facts(user_id: str, transcript: str, limit: int) -> list[str] | None:
    """Vector search over this user's facts, or None if unavailable."""
    if not db.vec_available():
        return None
    vectors = embeddings.embed([transcript])
    if not vectors:
        return None

    conn = db.connect()
    # The join is what turns row ids back into text. ORDER BY distance is
    # applied inside the vec0 table, so `facts` is only read for the winners.
    rows = conn.execute(
        """SELECT f.text
             FROM vec_facts v
             JOIN facts f ON f.id = v.fact_id
            WHERE v.user_id = ?
              AND v.embedding MATCH ?
              AND k = ?
         ORDER BY v.distance""",
        (user_id, embeddings.serialize(vectors[0]), limit),
    ).fetchall()
    return [row["text"] for row in rows]


def _describe(row) -> str:
    """'Лекция по матанализу, ауд. 305' -- title plus location when there is one."""
    return f"{row['title']}, {row['location']}" if row["location"] else row["title"]


def add_fact(user_id: str, text: str) -> int:
    """Store a new fact about the user. Returns its row id.

    This is the only write path into long-term memory, and it exists so the
    assistant's memory can grow by being spoken to rather than by a developer
    editing seed.py.

    Saying the same thing twice does not store it twice: people repeat
    themselves, and duplicates would burn the context budget on the same
    sentence twice over. The comparison is case-insensitive because STT
    capitalisation is not stable.

    The embedding is written in the same call, so a fact said out loud is
    searchable on the very next question rather than after a rebuild.
    """
    conn = db.connect()
    # Case folding happens in Python, not in SQL. SQLite's lower() is
    # ASCII-only: it leaves "Пьёт" untouched, so an SQL comparison would
    # treat it as different from "пьёт" and store the fact twice.
    wanted = text.strip().casefold()
    for row in conn.execute(
        "SELECT id, text FROM facts WHERE user_id = ?", (user_id,)
    ):
        if row["text"].strip().casefold() == wanted:
            return row["id"]

    cursor = conn.execute(
        "INSERT INTO facts (user_id, text) VALUES (?, ?)", (user_id, text)
    )
    fact_id = cursor.lastrowid
    _index_facts([(fact_id, user_id, text)])
    conn.commit()
    return fact_id


def reindex_facts() -> int:
    """Embed every fact that has no vector yet. Returns how many were done.

    Needed after seeding, and after copying the model onto a machine where
    facts were written while semantic search was unavailable.
    """
    if not db.vec_available():
        return 0
    rows = db.connect().execute(
        """SELECT f.id, f.user_id, f.text
             FROM facts f
             LEFT JOIN vec_facts v ON v.fact_id = f.id
            WHERE v.fact_id IS NULL"""
    ).fetchall()
    indexed = _index_facts([(r["id"], r["user_id"], r["text"]) for r in rows])
    db.connect().commit()
    return indexed


def _index_facts(items: list[tuple[int, str, str]]) -> int:
    """Embed and store vectors for (fact_id, user_id, text) triples."""
    if not items or not db.vec_available():
        return 0
    vectors = embeddings.embed([text for _, _, text in items])
    if not vectors:
        return 0

    conn = db.connect()
    conn.executemany(
        "INSERT INTO vec_facts (fact_id, user_id, embedding) VALUES (?, ?, ?)",
        [(fact_id, user_id, embeddings.serialize(vector))
         for (fact_id, user_id, _), vector in zip(items, vectors)],
    )
    return len(items)
