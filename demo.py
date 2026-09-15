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

from datetime import datetime

import jarvis_memory as memory

# Fixed point in time so the demo output does not drift from day to day.
NOW = datetime(2026, 9, 21, 14, 30)   # a Monday

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
    for user_id, transcript in CASES:
        show(user_id, transcript)

    # Multi-turn: shows that history really does end up in the messages
    # of the following request.
    print("#" * 70)
    print("# Multi-turn dialogue")
    print("#" * 70)
    memory.clear()
    memory.record_answer(
        "anton",
        "во сколько у меня лекция?",
        "В четыре часа дня, аудитория триста пять.",
        now=NOW,
    )
    show("anton", "а тренировка?")


if __name__ == "__main__":
    main()
