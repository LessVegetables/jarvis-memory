"""
Tests for the one language-model call module Ц makes on its own.

No server: llm_intent._ask is replaced with a function returning canned
replies, and a counter shows when it was (not) called. The two counter
assertions are the point of the design and not incidental -- that the model
is never asked about a phrase the regexes already understood, and that
everything still works when it cannot be reached at all.

Run:  python3 tests/test_llm_intent.py   (or python3 -m pytest tests/ -q)
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-test-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)

import jarvis_memory as memory  # noqa: E402
from jarvis_memory import db, history, llm_intent, router, seed  # noqa: E402

NOW = datetime(2026, 9, 14, 14, 30)


class FakeAsk:
    """Stands in for the HTTP call. `reply=None` is every kind of failure."""

    def __init__(self, reply=None):
        self.reply, self.calls, self.prompts = reply, 0, []

    def __call__(self, system, user, max_tokens=llm_intent.MAX_TOKENS):
        self.calls += 1
        self.prompts.append(user)
        return self.reply


def setup(ask=None, enabled="1"):
    os.environ["JARVIS_LLM_INTENT"] = enabled
    llm_intent._ask = ask if ask is not None else FakeAsk()
    db.close()
    seed.seed()
    history.clear()
    return llm_intent._ask


# --- parsing the reply --------------------------------------------------------

def test_parse_tolerates_what_a_small_model_actually_says():
    cases = [
        ("погода", router.WEATHER),
        ("Погода.", router.WEATHER),          # trailing punctuation
        ("метка: место", router.PLACES),      # a preamble it was told not to add
        ("расписание\n", router.SCHEDULE),
        ("ПОВТОРИ", router.REPEAT),
        ("зАпОмНи", router.REMEMBER),
        # The escape hatch is a label like any other; it just means "no intent".
        ("другое", None),
        # Two labels is not a classification. Guessing which one it meant is
        # how a wrong answer gets stated confidently.
        ("погода место", None),
        # The prose escape the eval caught once, and anything else unparseable.
        ("не могу понять ваш запрос", None),
        ("", None),
    ]
    for reply, expected in cases:
        assert llm_intent._parse(reply) == expected, reply


# --- when it is and is not called ---------------------------------------------

def test_regex_hits_never_reach_the_model():
    """The whole reason there is no added latency on the common path."""
    ask = setup(FakeAsk("погода"))
    for phrase in ("какая сегодня погода?", "что у меня завтра?",
                   "до скольки работает аптека", "повтори",
                   "запомни, что у меня аллергия на орехи"):
        memory.build_context("daniil", phrase, now=NOW)
    assert ask.calls == 0, ask.prompts


def test_model_rescues_a_phrase_the_regexes_miss():
    ask = setup(FakeAsk("место"))
    ctx = memory.build_context("daniil", "мне бы кофе выпить где-нибудь", now=NOW)
    assert ask.calls == 1
    assert ctx.intent == router.PLACES


def test_unreachable_model_falls_back_to_general():
    """A dead server, a timeout and an unparseable answer are the same thing:
    the intent the caller would have used anyway."""
    for reply in (None, "не могу понять ваш запрос", "погода место", "другое"):
        ask = setup(FakeAsk(reply))
        ctx = memory.build_context("daniil", "мне бы кофе выпить где-нибудь", now=NOW)
        assert ask.calls == 1, reply
        assert ctx.intent == router.GENERAL, (reply, ctx.intent)


def test_kill_switch_stops_the_call_entirely():
    ask = setup(FakeAsk("место"), enabled="0")
    ctx = memory.build_context("daniil", "мне бы кофе выпить где-нибудь", now=NOW)
    assert ask.calls == 0
    assert ctx.intent == router.GENERAL


def test_rescued_intent_becomes_the_sticky_one():
    """A rescued turn has to support a follow-up like any other, or "а
    завтра?" after it falls back to GENERAL again."""
    ask = setup(FakeAsk("расписание"))
    memory.build_context("daniil", "чем я буду занят", now=NOW)
    assert ask.calls == 1
    ctx = memory.build_context("daniil", "а завтра?", now=NOW)
    assert ctx.intent == router.SCHEDULE
    assert ask.calls == 1, "the follow-up was already resolved by the router"


TESTS = [
    test_parse_tolerates_what_a_small_model_actually_says,
    test_regex_hits_never_reach_the_model,
    test_model_rescues_a_phrase_the_regexes_miss,
    test_unreachable_model_falls_back_to_general,
    test_kill_switch_stops_the_call_entirely,
    test_rescued_intent_becomes_the_sticky_one,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} llm_intent tests")
