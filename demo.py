#!/usr/bin/env python3
"""
Memory module demo: prints the prompts that would go to the LLM.

Run:  python3 demo.py

The point of this file is to make the module's work visible. A prompt should
never be a black box: when the model answers nonsense, the first question is
always "what exactly did it get as input", and answering that should take one
command.

The test phrases are Russian because that is what users will say.
"""

from datetime import date, datetime, time, timedelta

import jarvis_memory as memory
from jarvis_memory import db, embeddings, seed

# Always a Monday at 14:30, computed from today rather than hardcoded, so it
# lines up with the events the seed script generates for the current week.
_TODAY = date.today()
NOW = datetime.combine(_TODAY - timedelta(days=_TODAY.weekday()), time(14, 30))

CASES = [
    # The same question, four times. Two invented schedules that differ, one
    # real one, and a guest who is refused -- that difference is the module.
    ("daniil", "какое у меня сегодня расписание?"),
    ("fedor", "какое у меня сегодня расписание?"),
    # Seeded with no events by design (seed.CALENDAR_ONLY), so this block is
    # empty until tools/sync_calendars.py has run against a real calendar.
    # An empty block here means sync has not run, not that the code is broken.
    ("daniel", "какое у меня сегодня расписание?"),
    (None,    "какое у меня сегодня расписание?"),
    ("daniil", "что приготовить на ужин?"),
    (None,    "какая сегодня погода?"),

    # Places: the same search, five different questions. Each block names one
    # business and carries only the fields that answer what was asked -- the
    # thing to look at here is what is NOT in each prompt.
    ("daniil", "какая аптека ближе всего"),
    ("daniil", "какая аптека сейчас работает"),
    ("daniil", "до скольки работает аптека"),
    ("daniil", "у какой аптеки рейтинг лучше"),
    ("daniil", "какие аптеки рядом"),

    # A preference that changes behaviour: after this, Экона stops appearing
    # in the answer above -- and the sentence itself stops appearing too, so
    # the name is not reintroduced into the prompt it was filtered out of.
    ("daniil", "запомни, я терпеть не могу аптеку Экона"),
    ("daniil", "какая аптека ближе всего"),

    # Music: the command has already run by the time the model sees this.
    ("daniil", "включи мой любимый плейлист"),
]


def show(user_id, transcript):
    ctx = memory.build_context(user_id, transcript, now=NOW)
    print("=" * 70)
    print(f"user_id={user_id!r}  intent={ctx.intent}  ~{ctx.estimate_tokens()} tokens")
    print("-" * 70)
    for message in ctx.to_messages():
        print(f"[{message['role']}]")
        print(message["content"])
    print()


def main():
    if not db.is_seeded():
        print(f"Empty database, seeding {db.db_path()}\n")
        seed.seed()

    # Say which retrieval path is running. Without this line the fallback is
    # invisible, and "the facts look wrong" is impossible to diagnose.
    if db.vec_available() and embeddings.is_available():
        print("Fact lookup: semantic search (sqlite-vec + embedding model)\n")
    else:
        missing = "no embedding model" if db.vec_available() else "no sqlite-vec"
        print(f"Fact lookup: most-recent fallback ({missing}). "
              f"See tools/export_embedding_model.py\n")

    for user_id, transcript in CASES:
        show(user_id, transcript)

    # Multi-turn: history ends up in the messages of the next request, and
    # the short follow-up "а завтра?" stays on the schedule topic instead of
    # falling through to GENERAL.
    print("#" * 70)
    print("# Multi-turn dialogue and sticky follow-up")
    print("#" * 70)
    memory.clear()
    show("daniil", "во сколько у меня лекция?")
    memory.record_answer(
        "daniil",
        "во сколько у меня лекция?",
        "В четыре часа дня, аудитория триста пять.",
        now=NOW,
    )
    show("daniil", "а завтра?")

    # "Повтори": the previous answer is handed back verbatim rather than
    # recomputed, so the model cannot drift into a different answer.
    print("#" * 70)
    print("# Repeat")
    print("#" * 70)
    show("daniil", "повтори, я не расслышал")

    # "Запомни, что...": the only path that writes. The fact is in the
    # database by the time the model is asked to confirm it.
    print("#" * 70)
    print("# Writing a new fact by voice")
    print("#" * 70)
    show("daniil", "запомни, что я не ем острое")
    show(None, "запомни, что я не ем острое")

    # Recalling a conversation from days ago. Needs the embedding model:
    # past dialogue has no non-semantic fallback, since random old chatter
    # is worse than nothing.
    print("#" * 70)
    print("# Recalling a past conversation (needs the embedding model)")
    print("#" * 70)
    memory.clear()
    memory.record_answer(
        "daniil",
        "Бобик что-то приболел, что делать?",
        "Свози Бобика к ветеринару на Ленина, он открыт до восьми.",
        now=NOW - timedelta(days=3),
    )
    show("daniil", "как думаешь, Бобику уже лучше?")


if __name__ == "__main__":
    main()
