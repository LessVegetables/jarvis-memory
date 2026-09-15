"""
Context assembly -- the core of module C.

There is not a single language-model call in this file. Everything
build_context does is a handful of data lookups and some string substitution.
The one LLM inference in the whole chain happens later and elsewhere: the
orchestrator takes the finished messages and hands them to Qwen exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from . import history, prompts, router, store

_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг",
             "пятница", "суббота", "воскресенье"]


@dataclass
class PromptContext:
    """Everything needed for one request to the LLM."""

    system_prompt: str
    history: list[dict[str, str]]
    transcript: str
    user_id: str | None
    intent: str
    debug: dict = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """The full message list -- what the orchestrator passes to the model."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": self.transcript},
        ]

    def estimate_tokens(self) -> int:
        """Rough budget estimate: Russian text runs about 2.5 chars per token.

        Precision is not the point here, order of magnitude is: if this creeps
        towards a thousand, history and data blocks are about to start
        crowding each other out.
        """
        chars = sum(len(m["content"]) for m in self.to_messages())
        return int(chars / 2.5)


def build_context(user_id: str | None, transcript: str,
                  now: datetime | None = None) -> PromptContext:
    """Assemble context from (who asked, what they asked).

    user_id=None means module B did not recognise the speaker.
    """
    now = now or datetime.now()
    intent = router.route(transcript)
    profile = store.get_profile(user_id) if user_id else None

    # No profile means there is no identified user, whatever the caller
    # passed as user_id. Enforced here rather than trusted from upstream.
    if profile is None:
        who = prompts.WHO_UNKNOWN
        blocks: list[str] = []
    else:
        who = prompts.WHO_KNOWN.format(name=profile["name"])
        blocks = _blocks_for(user_id, transcript, intent, now.date())

    system_prompt = prompts.SYSTEM_TEMPLATE.format(
        who=who,
        now=_format_now(now),
        blocks=("\n" + "\n\n".join(blocks)) if blocks else "",
    )

    ctx = PromptContext(
        system_prompt=system_prompt,
        history=history.get_history(user_id, now=now),
        transcript=transcript,
        user_id=user_id,
        intent=intent,
        debug={"profile_found": profile is not None, "blocks": len(blocks)},
    )
    ctx.debug["est_tokens"] = ctx.estimate_tokens()
    return ctx


def _blocks_for(user_id: str, transcript: str,
                intent: str, today: date) -> list[str]:
    """Which data blocks belong in the prompt for this intent.

    Key decision: blocks are selective. A question about the weather should
    not drag the schedule into context. That is partly token thrift, but
    mostly because a small model grabs at whatever is in front of it and
    answers the wrong question.
    """
    blocks = []

    if intent == router.SCHEDULE:
        events = store.get_schedule(user_id, today)
        if events:
            lines = "\n".join(f"- {time} {title}" for time, title in events)
            blocks.append(prompts.SCHEDULE_BLOCK.format(lines=lines))
        else:
            blocks.append(prompts.SCHEDULE_EMPTY)

    # Facts help where the answer depends on someone's preferences,
    # and are dead weight in a question about the weather.
    if intent in (router.PLACES, router.GENERAL):
        facts = store.get_facts(user_id, transcript)
        if facts:
            lines = "\n".join(f"- {fact}" for fact in facts)
            blocks.append(prompts.FACTS_BLOCK.format(lines=lines))

    # STEP 6: a weather block and a 2GIS block go here -- those will be HTTP
    # requests made at question time, not data read from the database.

    return blocks


def _format_now(now: datetime) -> str:
    return f"{_WEEKDAYS[now.weekday()]}, {now.day:02d}.{now.month:02d}, {now:%H:%M}"
