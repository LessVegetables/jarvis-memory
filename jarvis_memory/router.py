"""
Intent detection from the text of the question.

Why regexes rather than letting the LLM pick a tool (function calling):
Qwen2.5-1.5B quantised to w8a8, prompted in Russian, chooses tools
unreliably -- and it costs a whole extra inference pass to do it. In a
wake word -> STT -> LLM -> TTS chain, every extra pass is real seconds of
latency on the board. These patterns run in microseconds, can be debugged
by reading them, and fail predictably.

The patterns are Russian because the user speaks Russian. They are matched
against lowercased text with the letter ё folded to е, since STT output is
inconsistent about it.

Adding an intent: add the constant, add its patterns, add rows to
tests/test_router.py. The test table is the specification of what the
assistant understands.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import num_to_words

SCHEDULE = "schedule"
WEATHER = "weather"
REPEAT = "repeat"
REMEMBER = "remember"
PLACES = "places"
GENERAL = "general"     # nothing matched -- ordinary conversation

# Order matters: the first match wins. REMEMBER and REPEAT come first
# because they are commands about the conversation itself, and would
# otherwise be swallowed by a topic pattern -- "запомни, что завтра дождь"
# is a memory command, not a weather question.
_PATTERNS: list[tuple[str, str]] = [
    (REMEMBER, r"\bзапомни|\bзаметь\b|\bне забудь|\bзапиши\b"),
    (REPEAT, r"\bповтор|\bчто ты сказал|\bеще раз\b|\bне расслышал|\bчто-что\b"),
    (SCHEDULE, (r"\bраспис|\bпланы\b|\bчто у меня\b|\bво сколько у меня\b"
                r"|\bкогда у меня\b|\bчем я занят|\bсвободен\b|\bвстреч"
                r"|\bпар[ыа]\b|\bлекци|\bзаняти|\bтренировк")),
    (WEATHER, (r"\bпогод|\bдожд|\bтемператур|\bхолодно\b|\bтепло\b|\bжарко\b"
               r"|\bснег|\bветер|\bзонт|\bградус|\bпрогноз")),
    (PLACES, (r"\bрядом\b|\bпоблизости\b|\bработает до\b|\bкафе\b|\bресторан"
              r"|\bмагазин|\bаптек|\bдо скольки|\bоткрыт|\bзакрыт|\bкуда сходить"
              r"|\bгде купить|\bадрес\b")),
]

# A follow-up is a short phrase that matched nothing, e.g. "а тренировка?"
# after a schedule question. Without this the previous topic is lost and the
# model gets facts instead of the schedule it needs.
#
# Two ways to qualify, both deliberately strict. An earlier version also
# accepted "ends with a question mark", which hijacked "что приготовить на
# ужин?" into the previous schedule topic -- almost every question ends
# with one, so it is evidence of nothing. Being too eager here silently
# answers the wrong question, which is worse than losing the thread.
_FOLLOWUP_LEADING = r"^\s*(а|и|ну|ладно|хорошо|ок)\b"
_FOLLOWUP_MAX_WORDS = 4      # with a leading conjunction: "а что завтра?"

# Without a leading conjunction, only a bare continuation word counts.
# A word count alone is not enough: "расскажи анекдот" is two words and is
# plainly a new request, not a continuation. Telling those apart in general
# needs morphology (a verb in the imperative starts something new), which
# would mean another dependency; an explicit list is honest and costs nothing.
_FOLLOWUP_BARE = (r"^\s*(завтра|послезавтра|сегодня|вчера|позавчера"
                  r"|там|тут|здесь|потом)\s*\??\s*$")


def normalise(text: str) -> str:
    """Lowercase and fold ё to е. STT output is inconsistent about ё."""
    return text.lower().replace("ё", "е")


def route(transcript: str, previous_intent: str | None = None) -> str:
    """Return the intent for this phrase.

    `previous_intent` is the intent of the last turn, if it is still recent.
    It is used only to resolve short follow-ups; an explicit match always
    wins over it, so "а какая погода?" after a schedule question is still
    a weather question.
    """
    text = normalise(transcript)

    for intent, pattern in _PATTERNS:
        if re.search(pattern, text):
            return intent

    if previous_intent and previous_intent != GENERAL and _is_followup(text):
        return previous_intent

    return GENERAL


def _is_followup(text: str) -> bool:
    """Short phrase that looks like it continues the previous turn."""
    words = re.findall(r"\w+", text)
    if not words:
        return False
    if re.search(_FOLLOWUP_LEADING, text):
        return len(words) <= _FOLLOWUP_MAX_WORDS
    return bool(re.match(_FOLLOWUP_BARE, text))


def extract_fact(transcript: str) -> str:
    """Pull the fact out of "запомни, что у меня аллергия на орехи".

    Deliberately not an LLM call: this is a prefix strip, and routing it
    through the model would double the latency of every such request for
    no gain. Returns the original text if no trigger prefix is found.
    """
    stripped = re.sub(
        r"^\s*(пожалуйста[,\s]+)?(запомни|запиши|заметь|не забудь)"
        r"[,\s]*(что|пожалуйста)?[,\s]*",
        "",
        transcript,
        flags=re.IGNORECASE,
    )
    stripped = stripped.strip(" .,!")
    return stripped or transcript.strip()


# Which day a schedule question is about. Longest forms first: "послезавтра"
# must not be read as "завтра". Past days are here because "что у меня было
# вчера" is a perfectly ordinary question.
_DAY_OFFSETS = [
    (r"\bпозавчера\b", -2),
    (r"\bпослезавтра\b", 2),
    (r"\bвчера\b", -1),
    (r"\bзавтра\b", 1),
    (r"\bсегодня\b", 0),
]


WEEKDAYS = ["понедельник", "вторник", "среда", "четверг",
            "пятница", "суббота", "воскресенье"]
_RELATIVE_LABELS = {-2: "позавчера", -1: "вчера", 0: "сегодня",
                    1: "завтра", 2: "послезавтра"}


def day_label(day: date, today: date) -> str:
    """'завтра', or 'среда, семнадцатое сентября' for days further out.

    The label matters as much as the data under it: the model repeats
    whatever day-word it is given, so an unlabelled block invites a
    confident answer about the wrong day.
    """
    offset = (day - today).days
    if offset in _RELATIVE_LABELS:
        return _RELATIVE_LABELS[offset]
    return f"{WEEKDAYS[day.weekday()]}, {num_to_words.date_words(day)}"


def resolve_day(transcript: str, today: date) -> date:
    """Which day the question is about. Defaults to today.

    Without this, "а завтра?" returns today's events under a label saying
    "сегодня", and the model states the wrong day with full confidence.
    """
    text = normalise(transcript)
    for pattern, offset in _DAY_OFFSETS:
        if re.search(pattern, text):
            return today + timedelta(days=offset)
    return today
