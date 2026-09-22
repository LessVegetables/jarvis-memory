"""
Numbers to words, in Russian.

Why this exists: the assistant is listened to, not read. The system prompt
says so, but a 1.5B model takes the path of least resistance and echoes
whatever shape it sees in front of it -- give it "23.09, 01:44" and it will
say "23.09, 01:44" back, prompt rules notwithstanding. The fix is not a
sterner instruction, it is not putting digits in the prompt at all.

Two layers, and both are needed:

  * The named helpers (`date_words`, `time_words`, `count`, ...) are called
    at the point where the unit is known, so the numeral agrees with its
    noun: "одна минута" but "один метр", "три градуса" but "пять градусов".
  * `spell()` is a last pass over finished text -- facts the user dictated,
    calendar event titles, the transcript itself. Nothing there is under our
    control, so it is scrubbed rather than formatted.

Stands alone: standard library only, and it imports nothing from the rest
of the package, so any module here may use it without a cycle.

Nominative only, by design. Every caller phrases its output so that the
nominative fits; adding the other five cases would double this file to
serve sentences nobody writes. The one exception is `genitive`, which
exists because "с девяти до двадцати одного" has no nominative phrasing.

So `spell()` does get some agreement wrong in free text -- "с 2 командами"
becomes "с два командами" and not "с двумя командами". That is not a
regression: the phonemizer downstream reads a bare "2" as "два" and lands
in exactly the same place. Getting it right needs morphology of the
following word, which means a dependency this board does not have room for.
"""

from __future__ import annotations

import re
from datetime import date

__all__ = [
    "cardinal", "ordinal", "plural", "count", "genitive",
    "date_words", "year_words", "time_words", "time_range", "spell",
]

# --- cardinals -------------------------------------------------------------
# Gender only ever changes 1 and 2; the rest of the table is shared.
_UNITS = {
    "m": ["ноль", "один", "два", "три", "четыре",
          "пять", "шесть", "семь", "восемь", "девять"],
    "f": ["ноль", "одна", "две", "три", "четыре",
          "пять", "шесть", "семь", "восемь", "девять"],
    "n": ["ноль", "одно", "два", "три", "четыре",
          "пять", "шесть", "семь", "восемь", "девять"],
}
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
          "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок",
         "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста",
             "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

# "тысяча" is feminine -- "одна тысяча", not "один тысяча".
_SCALES = [
    ("тысяча", "тысячи", "тысяч", "f"),
    ("миллион", "миллиона", "миллионов", "m"),
    ("миллиард", "миллиарда", "миллиардов", "m"),
]


def plural(n: int, one: str, few: str, many: str) -> str:
    """Pick the noun form for a count: 1 год, 2 года, 5 лет.

    The 11-14 check comes first and is not an edge case to be clever about:
    without it, 11 takes the same form as 1 and the sentence is wrong.
    """
    n = abs(int(n))
    if n % 100 in (11, 12, 13, 14):
        return many
    rest = n % 10
    if rest == 1:
        return one
    if rest in (2, 3, 4):
        return few
    return many


def _triple(n: int, gender: str) -> list[str]:
    """Words for 1..999."""
    words = []
    if n >= 100:
        words.append(_HUNDREDS[n // 100])
        n %= 100
    if 10 <= n < 20:
        words.append(_TEENS[n - 10])
        n = 0
    elif n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        words.append(_UNITS[gender][n])
    return words


def cardinal(n: int, gender: str = "m") -> str:
    """42 -> 'сорок два'. `gender` is that of the noun being counted."""
    n = int(n)
    if n < 0:
        return "минус " + cardinal(-n, gender)
    if n == 0:
        return "ноль"

    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000

    words: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        if index == 0:
            words += _triple(group, gender)
        else:
            one, few, many, scale_gender = _SCALES[index - 1]
            words += _triple(group, scale_gender)
            words.append(plural(group, one, few, many))
    return " ".join(words)


def count(n, forms: tuple[str, str, str], gender: str = "m") -> str:
    """Number plus its noun, agreeing: count(2, ('час','часа','часов'))."""
    return f"{cardinal(n, gender)} {plural(n, *forms)}"


# --- ordinals --------------------------------------------------------------
# Used for dates ("двадцать третье" -- число is neuter) and years. Only the
# forms that actually occur are tabulated; a general ordinal generator would
# be more code and no more correct.
_ORDINALS = {
    1: ("первый", "первая", "первое"),
    2: ("второй", "вторая", "второе"),
    3: ("третий", "третья", "третье"),
    4: ("четвёртый", "четвёртая", "четвёртое"),
    5: ("пятый", "пятая", "пятое"),
    6: ("шестой", "шестая", "шестое"),
    7: ("седьмой", "седьмая", "седьмое"),
    8: ("восьмой", "восьмая", "восьмое"),
    9: ("девятый", "девятая", "девятое"),
    10: ("десятый", "десятая", "десятое"),
    11: ("одиннадцатый", "одиннадцатая", "одиннадцатое"),
    12: ("двенадцатый", "двенадцатая", "двенадцатое"),
    13: ("тринадцатый", "тринадцатая", "тринадцатое"),
    14: ("четырнадцатый", "четырнадцатая", "четырнадцатое"),
    15: ("пятнадцатый", "пятнадцатая", "пятнадцатое"),
    16: ("шестнадцатый", "шестнадцатая", "шестнадцатое"),
    17: ("семнадцатый", "семнадцатая", "семнадцатое"),
    18: ("восемнадцатый", "восемнадцатая", "восемнадцатое"),
    19: ("девятнадцатый", "девятнадцатая", "девятнадцатое"),
    20: ("двадцатый", "двадцатая", "двадцатое"),
    30: ("тридцатый", "тридцатая", "тридцатое"),
}
_GENDERS = {"m": 0, "f": 1, "n": 2}


def ordinal(n: int, gender: str = "n") -> str:
    """23 -> 'двадцать третье'. Covers 1..39; falls back to the cardinal."""
    n = int(n)
    index = _GENDERS[gender]
    if n in _ORDINALS:
        return _ORDINALS[n][index]
    tens, rest = divmod(n, 10)
    if tens * 10 in _ORDINALS and rest in _ORDINALS:
        return f"{_TENS[tens]} {_ORDINALS[rest][index]}"
    return cardinal(n, gender)


# Masculine genitive ordinals -- only ever needed for "...шестого года".
_ORDINALS_GEN = {
    1: "первого", 2: "второго", 3: "третьего", 4: "четвёртого", 5: "пятого",
    6: "шестого", 7: "седьмого", 8: "восьмого", 9: "девятого", 10: "десятого",
    11: "одиннадцатого", 12: "двенадцатого", 13: "тринадцатого",
    14: "четырнадцатого", 15: "пятнадцатого", 16: "шестнадцатого",
    17: "семнадцатого", 18: "восемнадцатого", 19: "девятнадцатого",
    20: "двадцатого", 30: "тридцатого", 40: "сорокового", 50: "пятидесятого",
    60: "шестидесятого", 70: "семидесятого", 80: "восьмидесятого",
    90: "девяностого",
}


def year_words(year: int) -> str:
    """2026 -> 'две тысячи двадцать шестого года'."""
    year = int(year)
    rest = year % 100
    if rest == 0:                      # 2000, 1900 -- rare, not worth a table
        return f"{cardinal(year)} года"
    head = cardinal(year - rest) if year - rest else ""
    if rest in _ORDINALS_GEN:
        tail = _ORDINALS_GEN[rest]
    else:
        tens, unit = divmod(rest, 10)
        tail = f"{_TENS[tens]} {_ORDINALS_GEN[unit]}"
    return " ".join(filter(None, [head, tail, "года"]))


# --- genitive --------------------------------------------------------------
# Opening hours are the only phrasing with no nominative form: "с девяти до
# двадцати одного". Covers 0..59, which is all a clock needs.
_GEN_UNITS = ["нуля", "одного", "двух", "трёх", "четырёх",
              "пяти", "шести", "семи", "восьми", "девяти"]
_GEN_TEENS = ["десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати",
              "пятнадцати", "шестнадцати", "семнадцати", "восемнадцати",
              "девятнадцати"]
_GEN_TENS = ["", "", "двадцати", "тридцати", "сорока", "пятидесяти"]


def genitive(n: int) -> str:
    """0..59 in the genitive: 21 -> 'двадцати одного'."""
    n = int(n)
    if n < 10:
        return _GEN_UNITS[n]
    if n < 20:
        return _GEN_TEENS[n - 10]
    tens, rest = divmod(n, 10)
    head = _GEN_TENS[tens] if tens < len(_GEN_TENS) else cardinal(tens * 10)
    return head if not rest else f"{head} {_GEN_UNITS[rest]}"


# --- dates and clock times -------------------------------------------------
_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"]

_HOURS = ("час", "часа", "часов")
_MINUTES = ("минута", "минуты", "минут")


def date_words(day, month: int | None = None, year: int | None = None,
               with_year: bool = False) -> str:
    """date(2026, 9, 23) -> 'двадцать третье сентября'.

    The year is left out unless asked for. Every date the assistant speaks
    is within three days of now, so saying it aloud is noise the model would
    dutifully repeat.
    """
    if isinstance(day, date):
        day, month, year = day.day, day.month, (day.year if with_year else None)
    words = f"{ordinal(day, 'n')} {_MONTHS[int(month) - 1]}"
    return f"{words} {year_words(year)}" if year else words


def time_words(hours, minutes: int | None = None) -> str:
    """'01:44' -> 'час сорок четыре минуты'. A whole hour drops the minutes."""
    if minutes is None:
        hours, minutes = str(hours).split(":")
    hours, minutes = int(hours), int(minutes)
    # "один час" is what a clock reads out; a person says "час".
    spoken = "час" if hours == 1 else count(hours, _HOURS)
    if minutes == 0:
        return spoken
    return f"{spoken} {count(minutes, _MINUTES, 'f')}"


def _clock_genitive(value) -> str:
    hours, _, minutes = str(value).partition(":")
    words = genitive(int(hours))
    return f"{words} {genitive(int(minutes))}" if minutes and int(minutes) else words


def time_range(start, end) -> str:
    """('09:00', '21:00') -> 'с девяти до двадцати одного'."""
    return f"с {_clock_genitive(start)} до {_clock_genitive(end)}"


# --- the scrubber ----------------------------------------------------------
# Everything below serves spell(): text we did not format ourselves and
# cannot re-phrase -- dictated facts, calendar titles, STT output.

# (one, few, many, gender, suffix)
_UNIT_WORDS = {
    "км/ч": ("километр", "километра", "километров", "m", " в час"),
    "м/с": ("метр", "метра", "метров", "m", " в секунду"),
    "км": ("километр", "километра", "километров", "m", ""),
    "мм": ("миллиметр", "миллиметра", "миллиметров", "m", ""),
    "см": ("сантиметр", "сантиметра", "сантиметров", "m", ""),
    "мл": ("миллилитр", "миллилитра", "миллилитров", "m", ""),
    "мг": ("миллиграмм", "миллиграмма", "миллиграммов", "m", ""),
    "мин": ("минута", "минуты", "минут", "f", ""),
    "сек": ("секунда", "секунды", "секунд", "f", ""),
    "кг": ("килограмм", "килограмма", "килограммов", "m", ""),
    "руб": ("рубль", "рубля", "рублей", "m", ""),
    "₽": ("рубль", "рубля", "рублей", "m", ""),
    "%": ("процент", "процента", "процентов", "m", ""),
    "м": ("метр", "метра", "метров", "m", ""),
    "л": ("литр", "литра", "литров", "m", ""),
    "г": ("грамм", "грамма", "граммов", "m", ""),
    "ч": ("час", "часа", "часов", "m", ""),
}
_DEGREES = ("градус", "градуса", "градусов")
_FRACTIONS = {1: ("десятая", "десятых", "десятых"),
              2: ("сотая", "сотых", "сотых"),
              3: ("тысячная", "тысячных", "тысячных")}

_CLOCK = r"(?:[01]?\d|2[0-4]):[0-5]\d"
_RE_TIME_RANGE = re.compile(rf"\b({_CLOCK})\s*[–—-]\s*({_CLOCK})\b")
_RE_TIME = re.compile(rf"\b({_CLOCK})\b")
# Two-digit month required, so "1.5 км" is a measurement and not the 1st of May.
_RE_DATE = re.compile(r"\b(\d{1,2})\.(\d{2})(?:\.(\d{4}))?\b")
_RE_TEMPERATURE = re.compile(r"([+-]?\d+)\s*°\s*[CFСФ]?")
# Longest units first: the alternation is ordered, so "км" must precede "м".
_RE_UNIT = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*"
    r"(км/ч|м/с|км|мм|см|мл|мг|мин|сек|кг|руб\.?|₽|%|м|л|г|ч)"
    r"(?![а-яёa-z])", re.IGNORECASE)
_RE_DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
_RE_DIGIT_RUN = re.compile(r"\d{7,}")
_RE_INTEGER = re.compile(r"\d+")
_LEFTOVERS = [("°", " градусов"), ("%", " процентов"), ("№", "номер "),
              ("&", " и "), ("=", " равно ")]
_RE_GAP = re.compile(r"[ \t]{2,}")
# A digit touching a letter: "6к1" would otherwise come out "шестькодин",
# and "10км" would never be recognised as a measurement.
_RE_GLUED = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
# "10/2" -- a house number or a fraction. Either way it is read "дробь",
# and the slash itself must not survive into the speech synthesiser.
_RE_SLASH = re.compile(r"(\d)\s*/\s*(\d)")


def _decimal_words(whole: str, fraction: str) -> str:
    """'1,5' -> 'одна целая пять десятых'."""
    forms = _FRACTIONS.get(len(fraction))
    if forms is None:                  # more precision than anyone says aloud
        return f"{cardinal(int(whole))} {cardinal(int(fraction))}"
    return (f"{count(int(whole), ('целая', 'целых', 'целых'), 'f')} "
            f"{count(int(fraction), forms, 'f')}")


def _spell_date(match: re.Match) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group(0)          # not a date after all -- leave it
    return date_words(day, month, int(year) if year else None)


def _spell_temperature(match: re.Match) -> str:
    raw = match.group(1)
    words = count(int(raw), _DEGREES)
    return f"плюс {words}" if raw.startswith("+") else words


def _spell_unit(match: re.Match) -> str:
    number, unit = match.group(1), match.group(2).lower().rstrip(".")
    one, few, many, gender, suffix = _UNIT_WORDS[unit]
    if "." in number or "," in number:
        whole, _, fraction = number.replace(",", ".").partition(".")
        # A fraction takes the genitive singular: "полтора километра".
        return f"{_decimal_words(whole, fraction)} {few}{suffix}"
    return count(int(number), (one, few, many), gender) + suffix


def spell(text: str) -> str:
    """Rewrite every digit in `text` as words. Idempotent, and never raises.

    Applied to finished text, so it guesses: "23.09" is a date, "36%" is a
    percentage, a seven-digit run is a phone number and gets read out digit
    by digit. Where the caller knows better it should use the helpers above
    and this pass then finds nothing left to do.
    """
    if not text:
        return text
    try:
        # First, because it is what makes "10км" a measurement and not a word.
        text = _RE_GLUED.sub(" ", text)
        text = _RE_TIME_RANGE.sub(
            lambda m: time_range(m.group(1), m.group(2)), text)
        text = _RE_TIME.sub(lambda m: time_words(m.group(1)), text)
        text = _RE_DATE.sub(_spell_date, text)
        text = _RE_TEMPERATURE.sub(_spell_temperature, text)
        text = _RE_UNIT.sub(_spell_unit, text)
        text = _RE_SLASH.sub(r"\1 дробь \2", text)
        text = _RE_DECIMAL.sub(
            lambda m: _decimal_words(m.group(1), m.group(2)), text)
        text = _RE_DIGIT_RUN.sub(
            lambda m: " ".join(_UNITS["m"][int(d)] for d in m.group(0)), text)
        text = _RE_INTEGER.sub(lambda m: cardinal(int(m.group(0))), text)
        for symbol, word in _LEFTOVERS:
            text = text.replace(symbol, word)
    except Exception:                                          # noqa: BLE001
        # A mangled number is a bad sentence; a raised exception here is the
        # assistant losing its voice mid-answer. Strip digits and move on.
        return _RE_INTEGER.sub("", text)
    # Expansions such as "°" -> " градусов" leave doubled spaces behind.
    # Horizontal whitespace only: the prompt's line structure is load-bearing.
    return _RE_GAP.sub(" ", text)
