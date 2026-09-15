"""
Access to user data.

FOR NOW: hardcoded dicts, so the module can be handed to the orchestrator
today instead of blocking on the database.

STEP 2 of the plan: the same thing, but read from SQLite (jarvis.db). The
function signatures below will not change -- only their bodies will. So
app.py written against these functions today keeps working after the move
to a real database.

Seed values stay in Russian because they are the assistant's content.
"""

from __future__ import annotations

from datetime import date

# --- temporary data (STEP 2 replaces this with SQLite tables) ---------------

_PROFILES = {
    "anton": {"name": "Антон", "age": 21},
    "masha": {"name": "Маша", "age": 19},
}

# (weekday, time, title) -- weekday as in date.weekday(): 0 = Monday
_SCHEDULE = {
    "anton": [
        (0, "16:00", "Лекция по матанализу, ауд. 305"),
        (0, "19:30", "Тренировка"),
        (1, "10:00", "Лабораторная по физике"),
    ],
    "masha": [
        (0, "09:00", "Английский"),
        (0, "14:00", "Встреча с научруком"),
        (2, "18:00", "Репетиция"),
    ],
}

_FACTS = {
    "anton": ["Вегетарианец", "Аллергия на орехи", "Собаку зовут Бобик"],
    "masha": ["Не пьёт кофе после обеда", "Учится на втором курсе"],
}


# --- public functions (these signatures are the contract) -------------------

def get_profile(user_id: str) -> dict | None:
    """Return the user's profile, or None if no such user_id exists."""
    return _PROFILES.get(user_id)


def get_schedule(user_id: str, day: date) -> list[tuple[str, str]]:
    """Events for the given day: [(time, title), ...].

    This is a plain lookup by key, not a search by meaning. A schedule has
    rigid structure, so RAG would be not just unnecessary here but harmful:
    a SQL query either returns the right day or nothing. It cannot "almost
    guess" and hand back Wednesday.
    """
    rows = _SCHEDULE.get(user_id, [])
    return [(time, title) for weekday, time, title in rows if weekday == day.weekday()]


def get_facts(user_id: str, transcript: str, limit: int = 3) -> list[str]:
    """Facts about the user that are relevant to their question.

    FOR NOW: returns the first `limit` facts and ignores the question.
    STEP 5 of the plan: vector search goes here (sqlite-vec + embeddings)
    and `transcript` starts affecting the result. The argument exists
    already so that calling code will not have to change later.
    """
    return _FACTS.get(user_id, [])[:limit]
