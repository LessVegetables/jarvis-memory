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
from jarvis_memory import db, seed

# Always a Monday at 14:30, computed from today rather than hardcoded, so it
# lines up with the events the seed script generates for the current week.
_TODAY = date.today()
NOW = datetime.combine(_TODAY - timedelta(days=_TODAY.weekday()), time(14, 30))

CASES = [
    ("anton", "какое у меня сегодня расписание?"),
    ("masha", "какое у меня сегодня расписание?"),
    (None,    "какое у меня сегодня расписание?"),
    ("anton", "что приготовить на ужин?"),
    (None,    "какая сегодня погода?"),
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

    for user_id, transcript in CASES:
        show(user_id, transcript)

    # Multi-turn: history ends up in the messages of the next request, and
    # the short follow-up "а завтра?" stays on the schedule topic instead of
    # falling through to GENERAL.
    print("#" * 70)
    print("# Multi-turn dialogue and sticky follow-up")
    print("#" * 70)
    memory.clear()
    show("anton", "во сколько у меня лекция?")
    memory.record_answer(
        "anton",
        "во сколько у меня лекция?",
        "В четыре часа дня, аудитория триста пять.",
        now=NOW,
    )
    show("anton", "а завтра?")

    # "Повтори": the previous answer is handed back verbatim rather than
    # recomputed, so the model cannot drift into a different answer.
    print("#" * 70)
    print("# Repeat")
    print("#" * 70)
    show("anton", "повтори, я не расслышал")

    # "Запомни, что...": the only path that writes. The fact is in the
    # database by the time the model is asked to confirm it.
    print("#" * 70)
    print("# Writing a new fact by voice")
    print("#" * 70)
    show("anton", "запомни, что я не ем острое")
    show(None, "запомни, что я не ем острое")


if __name__ == "__main__":
    main()
