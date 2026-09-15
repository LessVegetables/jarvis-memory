#!/usr/bin/env python3
"""
Демонстрация модуля памяти: печатает промпты, которые уйдут в LLM.

Запуск:  python3 demo.py

Смысл этого файла — сделать работу модуля видимой. Промпт не должен быть
чёрным ящиком: если модель отвечает ерунду, первый вопрос всегда "а что
именно она получила на вход", и ответ на него должен быть в одну команду.
"""

from datetime import datetime

import jarvis_memory as memory

# Фиксированный момент времени, чтобы вывод demo не менялся день ото дня.
NOW = datetime(2026, 9, 21, 14, 30)   # понедельник

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
    print(f"user_id={user_id!r}  intent={ctx.intent}  ~{ctx.estimate_tokens()} токенов")
    print("-" * 70)
    for message in ctx.to_messages():
        print(f"[{message['role']}]")
        print(message["content"])
    print()


def main():
    for user_id, transcript in CASES:
        show(user_id, transcript)

    # Диалог из нескольких реплик: показываем, что история реально
    # попадает в сообщения следующего запроса.
    print("#" * 70)
    print("# Многоходовый диалог")
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
