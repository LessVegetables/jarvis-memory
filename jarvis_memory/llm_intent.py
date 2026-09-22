"""
The one language-model call module Ц makes on its own.

router.py argues, at length and correctly, that regexes beat asking the model
to pick a tool: they run in microseconds, they can be debugged by reading
them, and they fail predictably. Nothing here contradicts that. This runs
only where the regexes have already returned GENERAL -- that is, only on
phrasings that matched nothing at all, where the alternative is not "a fast
right answer" but "no answer".

So the cost is paid exactly where there is nothing to lose:

  - Canonical phrasings never reach this file. No added latency on the path
    that already worked, and every row of the router's test table still
    passes by construction.
  - A failure here costs nothing either. The caller falls back to GENERAL,
    which is what it would have used anyway -- so a timeout, an unparseable
    answer, a dead server and a flatly wrong label all degrade to today's
    behaviour rather than to something worse.

Measured on the board: about 0.23 s, against 5-6 s of Whisper ahead of it.

--- what the prompt looks like the way it does -------------------------------

An eval over 52 utterances against Qwen2.5-0.5B on the board:

  - The dominant failure was a catch-all magnet. Whichever label the prompt
    made most salient swallowed everything: `general` was predicted 19 times
    for 9 true rows (42% precision) in the variant that both described it and
    repeated it in a closing sentence. In a variant that never mentioned it,
    the magnet simply moved to whatever was most prominent instead. Hence
    "другое" below: named once, last, with no description inviting it.
  - Dropping the escape hatch altogether is worse than the magnet. Off-domain
    speech -- "расскажи анекдот", "[музыка]", half-heard noise -- is most of
    what reaches this file, and with only real labels on offer the model must
    force-fit it into one. Today those correctly reach GENERAL, and a
    classifier that broke that would be a regression.
  - Labels are single Russian words, not English snake_case. The eval's
    hallucinated near-misses ("place_open_earliest" for place_opens_earliest,
    "day_now" for time_now) were all edit-distance artifacts of long compound
    English tokens. One Russian noun is a token or two and is anchored in the
    language the question was asked in.
  - Fine-grained distinctions stay out of this file. The place sub-intents
    were the eval's worst labels (44-50% recall); router.place_shape() scores
    25/26 on the same rows, deterministically and without the round trip.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from . import config, router

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_MODEL = "rkllm"

# Deliberately not the orchestrator's 120 s answer timeout. This call is an
# optimisation, and an optimisation that can hang the speaker for two minutes
# is not one. A slow server means GENERAL, immediately.
TIMEOUT = 1.5
# Enough for one word. The eval's one prose escape -- "не могу понять ваш
# запрос. Пожалуй" -- cannot happen inside five tokens.
MAX_TOKENS = 5

# None means "no intent": the caller keeps GENERAL.
_LABELS: dict[str, str | None] = {
    "погода": router.WEATHER,
    "место": router.PLACES,
    "расписание": router.SCHEDULE,
    "запомни": router.REMEMBER,
    "повтори": router.REPEAT,
    "другое": None,
}

PROMPT = """\
Определи, к какой теме относится фраза. Ответь одним словом из списка.

погода — погода, дождь, снег, температура, что надеть на улицу
место — магазины, аптеки, кафе, адреса, что находится и работает поблизости
расписание — планы человека, встречи, занятия, что у него сегодня или завтра
запомни — просьба запомнить что-то о человеке
повтори — просьба повторить сказанное
другое

Ответ — одно слово, без объяснений."""

_WORD = re.compile(r"[а-яё]+")


def enabled() -> bool:
    """JARVIS_LLM_INTENT=0 turns this off.

    Worth having: module Ц is one of five in a shared demo, and if the LLM
    server is unwell this is the piece that should be switched off first.
    Without the flag, doing that means editing code on the board.
    """
    return str(config.get("JARVIS_LLM_INTENT", "1")).strip().lower() not in (
        "0", "false", "no", "off")


def classify(transcript: str) -> str | None:
    """A router intent, or None if the model was no help. Never raises.

    Called only when router.route() has already returned GENERAL.
    """
    if not enabled() or not transcript.strip():
        return None
    reply = _ask(PROMPT, transcript)
    if reply is None:
        return None
    intent = _parse(reply)
    log.debug("llm intent: %r -> %r -> %r", transcript, reply, intent)
    return intent


def _parse(reply: str) -> str | None:
    """First known label word in the reply, if there is exactly one.

    Not a bare equality check: a small model asked for one word still says
    "метка: погода" or "погода." often enough to matter. Not a substring
    search either -- a reply naming two labels has not classified anything,
    and guessing which one it meant is how a wrong answer gets stated
    confidently.
    """
    found = [_LABELS[word] for word in _WORD.findall(reply.lower().replace("ё", "е"))
             if word in _LABELS]
    if len(found) != 1:
        return None
    return found[0]


def _ask(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str | None:
    """One completion from the same server the orchestrator talks to.

    urllib rather than requests, for the reason llm_module.py gives: the board
    has 20 GB of disk shared with other teams, and requests drags in urllib3,
    certifi and charset-normalizer for one POST to localhost.
    """
    url = config.get("JARVIS_LLM_URL", DEFAULT_URL)
    payload = json.dumps({
        "model": config.get("JARVIS_LLM_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or choice.get("text")
        return text if isinstance(text, str) else None
    except Exception:                                          # noqa: BLE001
        # Every failure is the same failure here: the caller keeps GENERAL.
        # Logged at debug because on a board with no LLM running this would
        # otherwise print a stack trace on every unrecognised phrase.
        log.debug("llm intent call failed", exc_info=True)
        return None
