"""
Fake data for development.

Run:  python3 -m jarvis_memory.seed

Two decisions worth explaining:

1. Events are generated RELATIVE TO TODAY, not written as fixed dates. A
   schedule seeded with hardcoded dates in September is empty by October,
   and the demo silently stops demonstrating anything. Re-running this
   script always produces a live two-week window.

2. There are ~8 facts per person, not 3. Step 5 (RAG) is about *choosing*
   which facts are relevant to a question. With three facts there is
   nothing to choose between and retrieval cannot be evaluated at all.

Data is Russian because it is the assistant's content.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from . import db

USERS = [
    ("anton", "Антон", 21),
    ("masha", "Маша", 19),
]

# (user_id, weekday, "HH:MM", title, location)
# weekday follows date.weekday(): 0 = Monday.
WEEKLY_EVENTS = [
    ("anton", 0, "16:00", "Лекция по матанализу", "ауд. 305"),
    ("anton", 0, "19:30", "Тренировка", "зал на Ленина"),
    ("anton", 1, "10:00", "Лабораторная по физике", "ауд. 112"),
    ("anton", 3, "14:00", "Семинар по программированию", "ауд. 401"),
    ("anton", 5, "12:00", "Созвон с командой проекта", None),
    ("masha", 0, "09:00", "Английский", "ауд. 208"),
    ("masha", 0, "14:00", "Встреча с научруком", "кафедра"),
    ("masha", 2, "18:00", "Репетиция", "актовый зал"),
    ("masha", 4, "11:00", "Зачёт по истории", "ауд. 305"),
]

FACTS = {
    "anton": [
        "Вегетарианец, мясо не ест",
        "Аллергия на орехи",
        "Собаку зовут Бобик",
        "Учится на третьем курсе, специальность — программная инженерия",
        "Живёт в комнате слева от кухни",
        "Не любит, когда его будят раньше девяти",
        "Пьёт кофе без сахара",
        "Занимается плаванием два раза в неделю",
    ],
    "masha": [
        "Не пьёт кофе после обеда",
        "Учится на втором курсе, факультет истории",
        "Играет на фортепиано",
        "Аллергии нет, но не ест острое",
        "Младшая сестра Антона",
        "Ложится спать рано, около одиннадцати",
        "Любит холодную погоду",
        "Копит на поездку в Петербург",
    ],
}


def seed(today: date | None = None) -> None:
    """Wipe and refill the database with development data."""
    today = today or date.today()
    conn = db.connect()

    # ON DELETE CASCADE clears events and facts along with the users.
    conn.execute("DELETE FROM users")

    conn.executemany("INSERT INTO users (user_id, name, age) VALUES (?, ?, ?)", USERS)

    monday = today - timedelta(days=today.weekday())
    rows = []
    for week in (0, 1):                     # this week and the next one
        for user_id, weekday, clock, title, location in WEEKLY_EVENTS:
            day = monday + timedelta(days=week * 7 + weekday)
            hour, minute = (int(part) for part in clock.split(":"))
            starts_at = datetime.combine(day, datetime.min.time()).replace(
                hour=hour, minute=minute
            )
            rows.append((user_id, starts_at.strftime("%Y-%m-%d %H:%M"), title, location))
    conn.executemany(
        "INSERT INTO events (user_id, starts_at, title, location) VALUES (?, ?, ?, ?)",
        rows,
    )

    conn.executemany(
        "INSERT INTO facts (user_id, text) VALUES (?, ?)",
        [(user_id, text) for user_id, texts in FACTS.items() for text in texts],
    )

    conn.commit()


def main() -> None:
    seed()
    conn = db.connect()
    counts = {
        table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("users", "events", "facts")
    }
    print(f"Seeded {db.db_path()}")
    for table, count in counts.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
