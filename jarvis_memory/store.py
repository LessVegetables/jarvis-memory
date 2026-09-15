"""
Access to user data, backed by SQLite.

The function signatures here are the contract the rest of the module is
written against. They did not change when the hardcoded dicts were replaced
by real queries in step 2, and they will not change again when fact lookup
becomes vector search in step 5. Only the bodies move.

Run `python3 -m jarvis_memory.seed` to fill the database with dev data.
"""

from __future__ import annotations

from datetime import date, datetime

from . import db, embeddings

# Per user. 500 exchanges is weeks of kitchen conversation, and at 312
# float32s each the vectors for it are ~600 KB. The cap is what keeps the
# archive from growing until the disk fills.
MAX_DIALOGUE_ROWS = 500


def get_profile(user_id: str) -> dict | None:
    """Return the user's profile, or None if no such user_id exists."""
    row = db.connect().execute(
        "SELECT user_id, name, age FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def add_user(user_id: str, name: str, age: int | None = None) -> None:
    """Create or update a user.

    user_id must be exactly what module B (speaker identification) emits
    for this person -- it is the key everything else hangs off. Nothing
    creates users implicitly: a calendar or a fact for an id that does not
    exist is a misconfiguration worth noticing, not something to paper over.
    """
    conn = db.connect()
    conn.execute(
        "INSERT INTO users (user_id, name, age) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET name = excluded.name, age = excluded.age",
        (user_id, name, age),
    )
    conn.commit()


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
        "SELECT starts_at, title, location, all_day FROM events "
        "WHERE user_id = ? AND date(starts_at) = ? "
        "ORDER BY all_day DESC, starts_at",
        (user_id, day.isoformat()),
    ).fetchall()
    return [("весь день" if row["all_day"] else row["starts_at"][11:16], _describe(row))
            for row in rows]


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


# --- dialogue archive --------------------------------------------------------

def archive_exchange(user_id: str, question: str, answer: str,
                     now: datetime | None = None) -> int:
    """Keep one exchange permanently, embedded so it can be recalled by meaning.

    Written immediately rather than when the turn ages out of the live
    buffer: simpler, and nothing is lost if the process dies in between.
    The overlap with the live buffer is handled on the read side, which
    excludes anything newer than the buffer's TTL.
    """
    now = now or datetime.now()
    conn = db.connect()
    cursor = conn.execute(
        "INSERT INTO dialogue (user_id, question, answer, created_at) "
        "VALUES (?, ?, ?, ?)",
        (user_id, question, answer, now.strftime("%Y-%m-%d %H:%M:%S")),
    )
    dialogue_id = cursor.lastrowid
    _index_dialogue([(dialogue_id, user_id, int(now.timestamp()),
                      _dialogue_document(question, answer))])
    _prune_dialogue(user_id)
    conn.commit()
    return dialogue_id


def get_dialogue(user_id: str, transcript: str, limit: int = 2,
                 before: datetime | None = None) -> list[tuple[str, str]]:
    """Past exchanges relevant to the question: [(question, answer), ...].

    `before` excludes turns newer than that moment -- the caller passes the
    live buffer's cutoff, since those turns are already in the messages and
    would otherwise appear twice.

    No non-semantic fallback here, unlike get_facts. Random old chatter is
    worse than nothing: the model will latch onto it. Without a model, the
    assistant simply does not recall past conversations.
    """
    if not db.vec_available():
        return []
    vectors = embeddings.embed([transcript])
    if not vectors:
        return []

    cutoff = int((before or datetime.now()).timestamp())
    rows = db.connect().execute(
        """SELECT d.question, d.answer
             FROM vec_dialogue v
             JOIN dialogue d ON d.id = v.dialogue_id
            WHERE v.user_id = ?
              AND v.created_at < ?
              AND v.embedding MATCH ?
              AND k = ?
         ORDER BY v.distance""",
        (user_id, cutoff, embeddings.serialize(vectors[0]), limit),
    ).fetchall()
    return [(row["question"], row["answer"]) for row in rows]


def reindex_dialogue() -> int:
    """Embed every archived exchange that has no vector yet."""
    if not db.vec_available():
        return 0
    rows = db.connect().execute(
        """SELECT d.id, d.user_id, d.question, d.answer, d.created_at
             FROM dialogue d
             LEFT JOIN vec_dialogue v ON v.dialogue_id = d.id
            WHERE v.dialogue_id IS NULL"""
    ).fetchall()
    items = [
        (r["id"], r["user_id"],
         int(datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S").timestamp()),
         _dialogue_document(r["question"], r["answer"]))
        for r in rows
    ]
    indexed = _index_dialogue(items)
    db.connect().commit()
    return indexed


def rebuild_vectors() -> dict[str, int]:
    """Drop every vector and embed everything again.

    Required whenever the embedding changes -- a different model, or a
    different pooling -- because vectors from two different embeddings are
    not comparable and mixing them makes search return nonsense.
    """
    if not db.vec_available():
        return {"facts": 0, "dialogue": 0}
    conn = db.connect()
    conn.execute("DELETE FROM vec_facts")
    conn.execute("DELETE FROM vec_dialogue")
    conn.commit()
    return {"facts": reindex_facts(), "dialogue": reindex_dialogue()}


def _dialogue_document(question: str, answer: str) -> str:
    """The text that gets embedded for one exchange -- both halves, so a
    later question can match either what was asked or what was answered."""
    return f"Вопрос: {question}\nОтвет: {answer}"


def _index_dialogue(items: list[tuple[int, str, int, str]]) -> int:
    """Embed and store (dialogue_id, user_id, created_at, document) rows."""
    if not items or not db.vec_available():
        return 0
    vectors = embeddings.embed([doc for _, _, _, doc in items])
    if not vectors:
        return 0
    db.connect().executemany(
        "INSERT INTO vec_dialogue (dialogue_id, user_id, created_at, embedding) "
        "VALUES (?, ?, ?, ?)",
        [(dialogue_id, user_id, created_at, embeddings.serialize(vector))
         for (dialogue_id, user_id, created_at, _), vector in zip(items, vectors)],
    )
    return len(items)


def _prune_dialogue(user_id: str) -> int:
    """Delete the oldest exchanges beyond MAX_DIALOGUE_ROWS. Returns how many."""
    conn = db.connect()
    total = conn.execute(
        "SELECT count(*) FROM dialogue WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    excess = total - MAX_DIALOGUE_ROWS
    if excess <= 0:
        return 0
    ids = [row["id"] for row in conn.execute(
        "SELECT id FROM dialogue WHERE user_id = ? ORDER BY created_at, id LIMIT ?",
        (user_id, excess),
    )]
    placeholders = ",".join("?" * len(ids))
    if db.vec_available():
        conn.execute(f"DELETE FROM vec_dialogue WHERE dialogue_id IN ({placeholders})", ids)
    conn.execute(f"DELETE FROM dialogue WHERE id IN ({placeholders})", ids)
    return len(ids)
