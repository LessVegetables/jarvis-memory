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

    # "А это далеко?" names no place, so nothing above matches it, and it is
    # not short enough to be a bare follow-up either. It is still obviously
    # about the place just discussed -- but only if a place was just
    # discussed, which is what the guard is for: with no such turn behind it,
    # "это далеко?" really is anyone's guess and GENERAL is the honest answer.
    if previous_intent == PLACES and re.search(_DISTANCE, text):
        return PLACES

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


# --- what shape of answer a question about places wants ---------------------
#
# The intent says the question is about places. This says what it wants to
# know, which decides both which of the found businesses gets into the prompt
# and which of its fields come with it.
#
# Why that matters more than it sounds: four businesses listed with four
# fields each is what produced "Аптека Экона, Морской проспект, шесть,
# восемьсот семьдесят метров" -- the name from the second listing, the address
# from the first, the distance from the third. The model was not inventing
# anything, it was reading across the rows of a table nobody told it was a
# table. Every extra field in the prompt is one more thing that can end up
# attached to the wrong name, so each shape below is rendered with only the
# fields that answer it.
PLACE_DISTANCE = "distance"              # "а это далеко?" -- about a place already named
PLACE_OPENS_EARLIEST = "opens_earliest"
PLACE_OPEN_NOW = "open_now"
PLACE_RATING = "rating"
PLACE_HOURS = "hours"
PLACE_ADDRESS = "address"
PLACE_LIST = "list"
PLACE_NEAREST = "nearest"                # the default: no qualifier means "which one"

# "далеко ли это", "сколько до неё идти" -- a distance question that names no
# place. Used twice: to pick the shape, and by route() to recognise the same
# phrase as a follow-up on a place named a turn ago.
_DISTANCE = (r"\bдалеко\b|\bсколько (до|метров|километров|идти)"
             r"|\bдолго (ли )?идти|\bблизко (ли )?(это|она|он)")

# First match wins, as in _PATTERNS. The order is the whole design here --
# the confusable phrasings all contain the keyword of a shape they do not
# mean, so the more specific question has to be asked first.
_PLACE_SHAPES: list[tuple[str, str]] = [
    # Checked first because it is the only one that needs no search at all:
    # it is about a place named in the previous turn.
    (PLACE_DISTANCE, _DISTANCE),

    # Before PLACE_NEAREST, because "во сколько откроется ближайшая аптека"
    # says "ближайшая" and is not asking which one is nearest.
    (PLACE_OPENS_EARLIEST, r"раньше всех|сам[аоы][яе] ранн|пораньше"
                           r"|раньше (всего|открыв|работает)"
                           r"|(работает|открывается) раньше"
                           r"|во сколько (она |он |оно )?(откр|начина)"
                           r"|когда откр"),

    # Deliberately not a bare "сейчас": "какая аптека сейчас ближе" is a
    # distance question that happens to contain the word.
    (PLACE_OPEN_NOW, r"сейчас (работает|открыт|закрыт)"
                     r"|(работает|открыт\w*|закрыт\w*) сейчас"
                     r"|прямо сейчас|еще (не )?(открыт|закрыл|работает)"
                     r"|не закрыл|есть открыт|уже открыл"
                     r"|\bоткрыт[аоы]?\b|\bзакрыт[аоы]?\b"),

    (PLACE_RATING, r"рейтинг|оценк|отзыв|лучш\w*|получше"
                   r"|хорош\w* (кафе|ресторан|аптек|магазин|мест)"),

    (PLACE_HOURS, r"до скольк|режим работы|часы работы|когда закрыв"
                  r"|во сколько закрыв|работает до|с[о]? скольки"),

    (PLACE_ADDRESS, r"\bадрес\b|где находится|на какой улице"
                    r"|как (туда )?(добраться|дойти|пройти)"),

    (PLACE_LIST, r"\bкакие\b|\bсписок\b|\bварианты\b|\bперечисли\b"
                 r"|что есть\b|что тут есть|что рядом есть"),
]


def place_shape(transcript: str) -> str:
    """Which shape of answer a places question wants.

    Defaults to PLACE_NEAREST: an unqualified "где аптека" is asking which
    one to walk to, and the nearest is the only honest answer to that.
    """
    text = normalise(transcript)
    for shape, pattern in _PLACE_SHAPES:
        if re.search(pattern, text):
            return shape
    return PLACE_NEAREST


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
