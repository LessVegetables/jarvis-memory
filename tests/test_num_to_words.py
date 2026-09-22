"""
Numeral spelling tests.

Like test_router.py, these tables are the specification: when the assistant
reads something out wrong, the fix is a row here first and a rule in
num_to_words.py second.

Run either way -- pytest if you have it, plain python if you do not:

    python3 -m pytest tests/ -q
    python3 tests/test_num_to_words.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import num_to_words as n  # noqa: E402

# (number, gender, expected)
CARDINAL_CASES = [
    (0, "m", "ноль"),
    (1, "m", "один"),
    (1, "f", "одна"),
    (2, "f", "две"),
    (11, "m", "одиннадцать"),
    (21, "f", "двадцать одна"),
    (44, "m", "сорок четыре"),
    (100, "m", "сто"),
    (305, "m", "триста пять"),
    (1000, "m", "одна тысяча"),
    (2026, "m", "две тысячи двадцать шесть"),
    (-3, "m", "минус три"),
]

# The 11-14 exception is the whole reason this function exists.
PLURAL_CASES = [
    (1, "год"), (2, "года"), (4, "года"), (5, "лет"),
    (11, "лет"), (12, "лет"), (14, "лет"),
    (21, "год"), (22, "года"), (25, "лет"),
    (101, "год"), (111, "лет"),
]

TIME_CASES = [
    ("01:44", "час сорок четыре минуты"),
    ("14:30", "четырнадцать часов тридцать минут"),
    ("21:00", "двадцать один час"),        # a whole hour drops the minutes
    ("00:05", "ноль часов пять минут"),
    ("09:01", "девять часов одна минута"),  # minutes are feminine
]

RANGE_CASES = [
    (("09:00", "21:00"), "с девяти до двадцати одного"),
    (("09:30", "18:00"), "с девяти тридцати до восемнадцати"),
    (("00:00", "24:00"), "с нуля до двадцати четырёх"),
]

DATE_CASES = [
    (date(2026, 9, 23), "двадцать третье сентября"),
    (date(2026, 9, 1), "первое сентября"),
    (date(2026, 1, 31), "тридцать первое января"),
    (date(2026, 5, 12), "двенадцатое мая"),
]

# The scrubber: text nobody formatted for us.
SPELL_CASES = [
    # the log that started this: date and clock out of the system prompt
    ("Сейчас: среда, 23.09, 01:44",
     "Сейчас: среда, двадцать третье сентября, час сорок четыре минуты"),
    # weather shapes
    ("+10°", "плюс десять градусов"),
    ("-3°", "минус три градуса"),
    ("ветер 10 км/ч", "ветер десять километров в час"),
    ("вероятность дождя 36%", "вероятность дождя тридцать шесть процентов"),
    # opening hours
    ("09:00–21:00", "с девяти до двадцати одного"),
    # a decimal must not be read as a date: 1.5 is not the 1st of May
    ("1.5 км", "одна целая пять десятых километра"),
    ("2,5 л", "две целых пять десятых литра"),
    # room numbers and other bare integers
    ("ауд. 305", "ауд. триста пять"),
    # a digit glued to a letter must not glue the words together
    ("дом 6к1", "дом шесть к один"),
    ("10км", "десять километров"),
    # a slash is a special symbol the synthesiser cannot say
    ("Мира 10/2", "Мира десять дробь два"),
    # long runs are phone numbers, read digit by digit
    ("звони 89991234567",
     "звони восемь девять девять девять один два три четыре пять шесть семь"),
    # a unit letter must not eat the start of a word
    ("5 минут", "пять минут"),
    ("через 2 часа", "через два часа"),
    # nothing to do
    ("никаких чисел здесь нет", "никаких чисел здесь нет"),
    ("", ""),
]


def test_cardinals():
    for number, gender, expected in CARDINAL_CASES:
        assert n.cardinal(number, gender) == expected, number


def test_plurals():
    for number, expected in PLURAL_CASES:
        assert n.plural(number, "год", "года", "лет") == expected, number


def test_times():
    for value, expected in TIME_CASES:
        assert n.time_words(value) == expected, value


def test_ranges():
    for (start, end), expected in RANGE_CASES:
        assert n.time_range(start, end) == expected, (start, end)


def test_dates():
    for day, expected in DATE_CASES:
        assert n.date_words(day) == expected, day
    assert n.date_words(date(2026, 9, 23), with_year=True) == (
        "двадцать третье сентября две тысячи двадцать шестого года")


def test_spell():
    failures = []
    for text, expected in SPELL_CASES:
        actual = n.spell(text)
        if actual != expected:
            failures.append(f"{text!r}: expected {expected!r}, got {actual!r}")
    assert not failures, "\n".join(failures)


def test_spell_is_idempotent():
    """to_messages() spells text the providers already spelled out. Running
    twice must change nothing, or that pass would corrupt every prompt."""
    for text, _ in SPELL_CASES:
        once = n.spell(text)
        assert n.spell(once) == once, text


def test_spell_leaves_no_digits():
    """The guarantee the orchestrator relies on."""
    samples = [text for text, _ in SPELL_CASES] + [
        "Пара 3 в 14:30, каб. 305, до 18:45",
        "купить 2 кг картошки за 150 руб",
        "день рождения 07.11.1999",
        "температура 36.6",
    ]
    for text in samples:
        assert not any(char.isdigit() for char in n.spell(text)), text


if __name__ == "__main__":
    test_cardinals()
    test_plurals()
    test_times()
    test_ranges()
    test_dates()
    test_spell()
    test_spell_is_idempotent()
    test_spell_leaves_no_digits()
    print(f"OK — {len(CARDINAL_CASES)} cardinals, {len(PLURAL_CASES)} plurals, "
          f"{len(TIME_CASES)} times, {len(RANGE_CASES)} ranges, "
          f"{len(DATE_CASES)} dates, {len(SPELL_CASES)} spelled")
