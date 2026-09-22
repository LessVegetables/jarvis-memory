"""
Fake data for development.

Run:  python3 -m jarvis_memory.seed

Four decisions worth explaining:

1. Events are generated RELATIVE TO TODAY, not written as fixed dates. A
   schedule seeded with hardcoded dates in September is empty by October,
   and the demo silently stops demonstrating anything. Re-running this
   script always produces a live two-week window.

2. There are ~8 facts per person, not 3. Step 5 (RAG) is about *choosing*
   which facts are relevant to a question. With three facts there is
   nothing to choose between and retrieval cannot be evaluated at all.

3. The people here are the team, and the point of the whole module is that
   "какое у меня расписание" answers differently depending on who asked.
   Five profiles that actually differ is the demo; two invented students
   were not.

4. `daniel` gets NO seeded events -- see CALENDAR_ONLY. His schedule comes
   from his real calendar through tools/sync_calendars.py, and a seeded row
   sitting next to a real one is the worst of both: it reads as real and it
   is not. Everyone else's events are deliberately absurd for the same
   reason, from the other direction -- nobody can mistake "подводный
   шахматный бокс" for a commitment they forgot about.

Data is Russian because it is the assistant's content.

A word of warning about `daniel` and `daniil`: two different people, one
letter apart, and SQLite will not complain about either. A typo in a
user_id does not fail -- it silently reads or writes the wrong person's
memory. Copy these strings, do not retype them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from . import db, store

# user_id must match what module B emits for that voice, which is the name
# of the folder under audio_module/audio_dataset/references/. `gleb` has no
# reference folder yet: his profile exists, so `--user-map` or a new folder
# will light it up, but until then his voice is a guest.
#
# Ages are deliberately None: these are real people and the assistant says
# the number out loud. Set your own with
#     python3 tools/add_user.py daniel "Даниэль" 21
USERS = [
    ("daniel", "Даниил", 22),
    ("daniil", "Даниил", None),
    ("fedor", "Фёдор", None),
    ("stepan", "Степан", None),
    ("gleb", "Глеб", None),
]

# Whose schedule is real and therefore not invented here. Enforced below
# rather than left to whoever edits WEEKLY_EVENTS next: an accidental seeded
# row for one of these users would be indistinguishable, in the answer, from
# a real appointment.
CALENDAR_ONLY = {"daniel"}

# (user_id, weekday, "HH:MM", title, location)
# weekday follows date.weekday(): 0 = Monday.
#
# Monday is loaded on purpose: demo.py fixes "now" to Monday 14:30, so a
# roster with nothing on Mondays demonstrates the empty-schedule branch and
# nothing else.
WEEKLY_EVENTS = [
    ("daniil", 0, "09:00", "Защита диплома перед комиссией из трёх котов", "ауд. 305"),
    ("daniil", 0, "16:00", "Лекция по матанализу", "ауд. 305"),
    ("daniil", 0, "19:30", "Тренировка по подводному шахматному боксу", "бассейн"),
    ("daniil", 1, "10:00", "Лабораторная: измерение скорости слухов", "ауд. 112"),
    ("daniil", 3, "14:00", "Семинар по программированию", "ауд. 401"),
    ("daniil", 5, "12:00", "Созвон с командой проекта", None),

    ("fedor", 0, "09:00", "Английский с профессором-попугаем", "ауд. 208"),
    ("fedor", 0, "14:00", "Встреча с научруком", "кафедра"),
    ("fedor", 2, "18:00", "Репетиция оркестра дрелей", "актовый зал"),
    ("fedor", 4, "11:00", "Зачёт по истории будущего", "ауд. 305"),

    ("stepan", 0, "11:00", "Калибровка гирлянды силой мысли", "лаба"),
    ("stepan", 0, "17:00", "Переговоры с поставщиком паяльников", None),
    ("stepan", 2, "13:00", "Пайка платы вслепую на скорость", "лаба"),
    ("stepan", 4, "20:00", "Ночной дозор у трёхмерного принтера", "лаба"),

    ("gleb", 0, "10:30", "Кастинг на роль голоса Джарвиса", "студия"),
    ("gleb", 0, "15:00", "Дегустация кофе из семи стран", "кухня"),
    ("gleb", 3, "09:00", "Утренняя пробежка за уезжающим автобусом", "Академгородок"),
    ("gleb", 6, "13:00", "Турнир по метанию бумажных самолётиков", "двор"),
]

# `daniel`'s facts are placeholders, not claims: mundane on purpose, because
# his profile is the one that is supposed to be real. Overwrite them by
# voice -- "запомни, что ..." writes straight into this table.
FACTS = {
    "daniel": [
        "Пьёт кофе без сахара",
        "Работает над голосовым ассистентом на плате",
        "Не любит, когда его будят раньше девяти",
        "Расписание берётся из его настоящего календаря",
        "Предпочитает короткие ответы без вступлений",
        "Живёт в Академгородке",
        "Ложится спать поздно",
        "Держит телефон на беззвучном режиме",
    ],
    "daniil": [
        "Вегетарианец, мясо не ест",
        "Аллергия на орехи",
        "Собаку зовут Бобик",
        "Учится на третьем курсе, специальность — программная инженерия",
        "Живёт в комнате слева от кухни",
        "Не любит, когда его будят раньше девяти",
        "Пьёт кофе без сахара",
        "Занимается плаванием два раза в неделю",
    ],
    "fedor": [
        "Не пьёт кофе после обеда",
        "Учится на втором курсе, факультет истории",
        "Играет на фортепиано",
        "Аллергии нет, но не ест острое",
        "Ложится спать рано, около одиннадцати",
        "Любит холодную погоду",
        "Копит на поездку в Петербург",
        "Ходит в бассейн по четвергам",
    ],
    "stepan": [
        "Отвечает за плату индикации на светодиодах",
        "Держит паяльник на столе справа от монитора",
        "Пьёт чай, кофе не переносит",
        "Аллергия на кошек",
        "Собирает механические клавиатуры",
        "Не ест грибы ни в каком виде",
        "Катается на сноуборде всю зиму",
        "Просит не звонить до полудня",
    ],
    "gleb": [
        "Записывал эталонные фразы для синтеза речи",
        "Разбирается в кофе лучше всех в команде",
        "Не ест сладкое",
        "Играет на гитаре",
        "Боится высоты, на крышу не пойдёт",
        "Ходит пешком везде, где меньше часа",
        "Учится на первом курсе",
        "Всегда носит с собой термос",
    ],
}


def seed(today: date | None = None) -> None:
    """Wipe and refill the database with development data.

    Note what this destroys: `DELETE FROM users` cascades, so a `daniel`
    whose events came from a real calendar loses them here too. They come
    back on the next `tools/sync_calendars.py` run, not on their own --
    main() says so out loud.
    """
    today = today or date.today()
    conn = db.connect()

    # ON DELETE CASCADE clears events and facts along with the users.
    conn.execute("DELETE FROM users")
    # vec_facts is a virtual table, so no foreign key reaches it. Left alone
    # it would keep vectors pointing at fact ids that no longer exist, and
    # searches would join against nothing and quietly return fewer results.
    if db.vec_available():
        conn.execute("DELETE FROM vec_facts")
        conn.execute("DELETE FROM vec_dialogue")

    conn.executemany("INSERT INTO users (user_id, name, age) VALUES (?, ?, ?)", USERS)

    monday = today - timedelta(days=today.weekday())
    rows = []
    for week in (0, 1):                     # this week and the next one
        for user_id, weekday, clock, title, location in WEEKLY_EVENTS:
            if user_id in CALENDAR_ONLY:
                continue                    # real calendar owns this schedule
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

    # Embed the freshly inserted facts, if a model is available. Without this
    # the facts exist but are invisible to semantic search.
    store.reindex_facts()


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

    if not db.vec_available():
        print("  vectors: none (sqlite-vec unavailable)")
    elif not embeddings_available():
        print("  vectors: none (no embedding model -- fact lookup will fall "
              "back to most-recent)")
    else:
        indexed = conn.execute("SELECT count(*) FROM vec_facts").fetchone()[0]
        print(f"  vectors: {indexed}")

    # Said every time, not only when it matters: the one run where this is
    # forgotten is the run where someone demos an empty schedule.
    print(f"\n  no seeded events for: {', '.join(sorted(CALENDAR_ONLY))}"
          " -- their schedule is real.")
    print("  run `python3 tools/sync_calendars.py` now to refill it.")


def embeddings_available() -> bool:
    from . import embeddings
    return embeddings.is_available()


if __name__ == "__main__":
    main()
