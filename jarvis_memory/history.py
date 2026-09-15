"""
Short-term dialogue memory: the last few exchanges, plus the last answer
(which is what makes "repeat" possible).

Kept in RAM rather than in the database: this is the state of the current
conversation, not a fact about a person. Losing it on reboot is fine.

Two constraints that did not exist in a Telegram bot backed by a cloud model:

1. MAX_TURNS is small. A local Qwen2.5-1.5B has a context of 2048 or 4096
   tokens -- the number is fixed when the model is converted to .rkllm and
   cannot be changed afterwards. History competes for that budget with the
   schedule and the facts, i.e. with the exact thing this project exists to
   show. On top of that, a small model holds the thread worse the longer the
   context gets: by the tenth turn it starts answering the sixth question.

2. TTL. A Telegram chat is a continuous private thread: come back a week
   later and the history is still yours and still relevant. A kitchen speaker
   is not a thread. If Anton asked something at 9am and walks up again at 7pm,
   replaying the morning is wrong -- that is a different conversation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

MAX_TURNS = 3          # exchanges (question + answer), not messages
TTL = timedelta(minutes=5)

# user_id -> [(timestamp, question, answer), ...]
_buffers: dict[str, list[tuple[datetime, str, str]]] = {}

# user_id -> (timestamp, intent) of the most recent routed question.
# Kept separately from _buffers because it is written when the question
# arrives, while the buffer entry only exists once the answer comes back.
# The router uses it to resolve short follow-ups like "а тренировка?".
_last_intent: dict[str, tuple[datetime, str]] = {}

# History for an unrecognised speaker is shared across all of them. Fine for
# a prototype, but worth remembering: two different guests would see each
# other's turns.
_UNKNOWN_KEY = "_unknown"


def _key(user_id: str | None) -> str:
    return user_id or _UNKNOWN_KEY


def record_answer(user_id: str | None, question: str, answer: str,
                  now: datetime | None = None) -> None:
    """Store one exchange. The orchestrator calls this after the LLM replies."""
    now = now or datetime.now()
    buf = _buffers.setdefault(_key(user_id), [])
    buf.append((now, question, answer))
    del buf[:-MAX_TURNS]


def get_history(user_id: str | None,
                now: datetime | None = None) -> list[dict[str, str]]:
    """Recent turns, fresher than TTL, in chat-message form.

    The [{"role": ..., "content": ...}] shape is not arbitrary: Qwen2.5-Instruct
    is trained on ChatML dialogue markup, and it understands past turns passed
    as separate user/assistant messages noticeably better than the same text
    glued into the system prompt as a "previously in this conversation"
    paragraph.
    """
    now = now or datetime.now()
    fresh = [(ts, q, a) for ts, q, a in _buffers.get(_key(user_id), [])
             if now - ts <= TTL]
    _buffers[_key(user_id)] = fresh

    messages = []
    for _, question, answer in fresh:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


def get_last_answer(user_id: str | None,
                    now: datetime | None = None) -> str | None:
    """Text of the most recent answer -- the basis for the "repeat" intent."""
    now = now or datetime.now()
    buf = _buffers.get(_key(user_id), [])
    if not buf:
        return None
    ts, _, answer = buf[-1]
    return answer if now - ts <= TTL else None


def set_last_intent(user_id: str | None, intent: str,
                    now: datetime | None = None) -> None:
    """Remember what this question was about, for follow-up resolution."""
    _last_intent[_key(user_id)] = (now or datetime.now(), intent)


def get_last_intent(user_id: str | None,
                    now: datetime | None = None) -> str | None:
    """Intent of the previous question, if it is still recent.

    Same TTL as the dialogue buffer: a follow-up to something asked this
    morning is not a follow-up, it is a new conversation.
    """
    now = now or datetime.now()
    entry = _last_intent.get(_key(user_id))
    if entry is None:
        return None
    timestamp, intent = entry
    return intent if now - timestamp <= TTL else None


def clear(user_id: str | None = None) -> None:
    """Drop history, for one user or all of them. Needed in tests."""
    if user_id is None:
        _buffers.clear()
        _last_intent.clear()
    else:
        _buffers.pop(_key(user_id), None)
        _last_intent.pop(_key(user_id), None)
