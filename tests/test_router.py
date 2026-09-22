"""
Router tests.

This table is the specification of what the assistant understands. When
someone reports "it didn't get me", the fix is a row here first, then a
pattern in router.py.

Run either way -- pytest if you have it, plain python if you do not:

    python3 -m pytest tests/ -q
    python3 tests/test_router.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import router  # noqa: E402

# (phrase, previous_intent, expected_intent)
CASES = [
    # --- schedule ---
    ("какое у меня сегодня расписание?", None, router.SCHEDULE),
    ("что у меня завтра?", None, router.SCHEDULE),
    ("во сколько у меня лекция?", None, router.SCHEDULE),
    ("когда у меня тренировка", None, router.SCHEDULE),
    ("я сегодня свободен?", None, router.SCHEDULE),

    # --- weather ---
    ("какая сегодня погода?", None, router.WEATHER),
    ("завтра будет дождь?", None, router.WEATHER),
    ("сколько градусов на улице", None, router.WEATHER),
    ("зонт нужен?", None, router.WEATHER),

    # --- places ---
    ("что есть поблизости?", None, router.PLACES),
    ("до скольки работает аптека", None, router.PLACES),
    ("где купить молоко", None, router.PLACES),

    # --- repeat ---
    ("повтори", None, router.REPEAT),
    ("повтори последнее", None, router.REPEAT),
    ("что ты сказал?", None, router.REPEAT),
    ("я не расслышал", None, router.REPEAT),

    # --- remember ---
    ("запомни, что у меня аллергия на орехи", None, router.REMEMBER),
    ("запиши что я не ем острое", None, router.REMEMBER),
    ("не забудь: я работаю по средам из дома", None, router.REMEMBER),

    # A memory command about a topic must stay a memory command. Without
    # REMEMBER being ordered before WEATHER, this would route to weather.
    ("запомни, что завтра дождь", None, router.REMEMBER),

    # --- general ---
    ("что приготовить на ужин?", None, router.GENERAL),
    ("расскажи анекдот", None, router.GENERAL),

    # --- follow-ups: the bug found in step 2's demo ---
    ("а завтра?", router.SCHEDULE, router.SCHEDULE),
    ("а завтра?", router.WEATHER, router.WEATHER),
    ("а там?", router.PLACES, router.PLACES),

    # An explicit match beats the sticky previous intent.
    ("а какая погода?", router.SCHEDULE, router.WEATHER),

    # Stickiness must not hijack a genuinely new short question just
    # because it is short and ends in "?". This one regressed once.
    ("что приготовить на ужин?", router.SCHEDULE, router.GENERAL),
    ("расскажи анекдот", router.SCHEDULE, router.GENERAL),
    # Bare one-word follow-up with no leading conjunction still works.
    ("завтра?", router.SCHEDULE, router.SCHEDULE),

    # Stickiness must not swallow a genuinely new, longer question.
    ("расскажи что-нибудь интересное про историю города",
     router.SCHEDULE, router.GENERAL),

    # Nothing to be sticky about.
    ("а завтра?", None, router.GENERAL),
    ("а завтра?", router.GENERAL, router.GENERAL),

    # --- distance follow-ups: "а это далеко?" names no place ---
    # Nothing in the phrase says it is about a place. What says so is that
    # the previous turn was, which is why the rule is guarded on it.
    ("а это далеко?", router.PLACES, router.PLACES),
    ("это далеко?", router.PLACES, router.PLACES),
    ("сколько до нее идти", router.PLACES, router.PLACES),
    ("сколько километров до нее", router.PLACES, router.PLACES),
    ("а сколько метров", router.PLACES, router.PLACES),

    # Without a place behind it the same phrase is anyone's guess, and
    # guessing PLACES would send it off to search 2GIS for nothing.
    ("это далеко?", None, router.GENERAL),
    ("сколько до нее идти", router.WEATHER, router.GENERAL),

    # --- ё folding: STT output is inconsistent about it ---
    ("ещё раз", None, router.REPEAT),
    ("еще раз", None, router.REPEAT),
]


def test_routing():
    failures = []
    for phrase, previous, expected in CASES:
        actual = router.route(phrase, previous)
        if actual != expected:
            failures.append(f"{phrase!r} (prev={previous}): "
                            f"expected {expected}, got {actual}")
    assert not failures, "\n".join(failures)


EXTRACTION_CASES = [
    ("запомни, что у меня аллергия на орехи", "у меня аллергия на орехи"),
    ("Запомни что я вегетарианец", "я вегетарианец"),
    ("запиши я не ем острое", "я не ем острое"),
    ("пожалуйста, запомни, что собаку зовут Бобик", "собаку зовут Бобик"),
    # No trigger word: return the text unchanged rather than an empty string.
    ("просто какая-то фраза", "просто какая-то фраза"),
]


def test_fact_extraction():
    for phrase, expected in EXTRACTION_CASES:
        assert router.extract_fact(phrase) == expected, phrase


if __name__ == "__main__":
    test_routing()
    test_fact_extraction()
    print(f"OK — {len(CASES)} routing cases, "
          f"{len(EXTRACTION_CASES)} extraction cases")
