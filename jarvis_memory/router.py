"""
Intent detection from the text of the question.

THIS IS A STUB. The real router is step 4 of the plan; what is here is just
enough for demo.py to show different branches. Do not grow this file until
steps 2-3 (database and the SQL path) are done.

Why not let the LLM do function calling instead: Qwen2.5-1.5B, quantised to
w8a8, prompted in Russian, picks tools unreliably -- and it costs a whole
extra inference pass to do it. In a wake word -> STT -> LLM -> TTS chain,
every extra pass is real seconds of latency on the board. Thirty regexes run
in microseconds, can be debugged by reading them, and fail predictably.

The patterns are Russian because the user speaks Russian.
"""

from __future__ import annotations

import re

SCHEDULE = "schedule"
WEATHER = "weather"
REPEAT = "repeat"
PLACES = "places"
GENERAL = "general"     # nothing matched -- ordinary conversation

_PATTERNS = [
    (REPEAT, r"\bповтор|\bчто ты сказал|\bещё раз\b"),
    (SCHEDULE, r"\bраспис|\bпланы\b|\bчто у меня\b|\bво сколько у меня\b"),
    (WEATHER, r"\bпогод|\bдожд|\bтемператур|\bхолодно\b|\bтепло\b"),
    (PLACES, r"\bрядом\b|\bпоблизости\b|\bработает до\b|\bкафе\b|\bресторан"),
]


def route(transcript: str) -> str:
    """Return the intent. First match wins, so the order above matters."""
    text = transcript.lower()
    for intent, pattern in _PATTERNS:
        if re.search(pattern, text):
            return intent
    return GENERAL
