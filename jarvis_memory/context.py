"""
Context assembly -- the core of module C.

This file used to say it contained no language-model call at all. That is no
longer true, and the exception is worth stating precisely, because the rule it
breaks was a good one.

Everything build_context does is still data lookup and string substitution.
The model is never asked what the answer is, never asked to choose between
records, and never asked to compare anything: providers and store decide, the
prompt states, and the one inference the orchestrator runs afterwards only
puts the finished facts into a sentence.

The exception is deciding what was *asked*. When router.route() matches
nothing -- not the topic patterns, not the follow-up rules -- the alternative
to asking the model is GENERAL, which means the question gets answered with
no data at all. So on that path, and only on that path, llm_intent.classify()
gets a turn. It is a rescue attempt on input that has already failed
everything cheaper; if it fails too, the result is the GENERAL the caller
would have used anyway. See llm_intent.py for why it is shaped the way it is.

So the honest version of the old sentence: one language-model call, on the
path where there was nothing to lose, and never to decide an answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from . import history, llm_intent, num_to_words, prompts, router, store
from .providers import places, spotify, weather

log = logging.getLogger(__name__)

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
        """The full message list -- what the orchestrator passes to the model.

        Every message goes through `num_to_words.spell` on the way out, and
        that is the module's one hard guarantee: not a single digit reaches
        the model. Telling a 1.5B model to say numbers as words does not
        work -- it copies the shape it is shown, so "01:44" in the prompt
        comes back as "01:44" in the answer and the speaker reads it out as
        a string of characters. The blocks above are already spelled out by
        the code that built them; this pass catches what we do not write
        ourselves: dictated facts, calendar titles, the transcript.

        It is idempotent, so spelled-out text passing through again is a
        no-op -- a few hundred characters of regex against a two-second
        inference.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": self.transcript},
        ]
        return [{"role": m["role"], "content": num_to_words.spell(m["content"])}
                for m in messages]

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
    # Route with the previous turn's intent in hand so that a short follow-up
    # ("а тренировка?") stays on the previous topic instead of falling
    # through to GENERAL, then record the new intent for the next turn.
    intent = router.route(transcript, history.get_last_intent(user_id, now=now))
    if intent == router.GENERAL:
        # Nothing matched -- not a topic, not a follow-up. Ask the model
        # before giving up; GENERAL is what we keep if it cannot help.
        intent = llm_intent.classify(transcript) or router.GENERAL
    history.set_last_intent(user_id, intent, now=now)
    profile = store.get_profile(user_id) if user_id else None

    # No profile means there is no identified user, whatever the caller
    # passed as user_id. Enforced here rather than trusted from upstream.
    # Public blocks first: anyone in the kitchen may ask about the weather
    # or what is nearby, recognised or not.
    blocks: list[str] = []
    if intent == router.MUSIC:
        # Like REMEMBER below, this one acts before the model speaks: the
        # command has already run by the time the block is written, and the
        # block says what happened.
        blocks.append(spotify.block(transcript, now))
    elif intent == router.WEATHER:
        blocks.append(weather.block(transcript, now))
    elif intent == router.PLACES:
        # The answer carries the one place it named, if it named one. Holding
        # onto it is what makes the next turn's "а это далеко?" answerable
        # about that business rather than about a fresh search's first hit.
        found = places.answer(
            transcript, now,
            last_place=history.get_last_place(user_id, now=now),
            # Applied before anything is selected, so a place someone asked
            # not to be offered cannot come back as "the nearest" either.
            dislikes=store.get_dislikes(user_id) if user_id else ())
        blocks.append(found.text)
        if found.place is not None:
            history.set_last_place(user_id, found.place, now=now)

    if profile is None:
        # Personal intents get the refusal. Anything else is answered as to
        # a guest -- refusing to say what the weather is would be absurd.
        who = (prompts.WHO_UNKNOWN
               if intent in (router.SCHEDULE, router.REMEMBER)
               else prompts.WHO_GUEST)
    else:
        if profile["age"] is None:
            who = prompts.WHO_KNOWN.format(name=profile["name"])
        else:
            who = prompts.WHO_KNOWN_AGE.format(
                name=profile["name"],
                age=num_to_words.count(profile["age"], ("год", "года", "лет")),
            )
        blocks += _blocks_for(user_id, transcript, intent, now)

    system_prompt = num_to_words.spell(prompts.SYSTEM_TEMPLATE.format(
        who=who,
        now=_format_now(now),
        blocks=("\n" + "\n\n".join(blocks)) if blocks else "",
    ))

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


def record_answer(user_id: str | None, question: str, answer: str,
                  now: datetime | None = None) -> None:
    """Feed the model's answer back into memory. The orchestrator calls this
    after every reply.

    Two destinations: the live buffer in RAM (what the next few turns see)
    and the permanent archive in SQLite (what can be recalled weeks later).
    Unrecognised speakers get the first and not the second -- a guest's
    conversation is nobody's memory.
    """
    now = now or datetime.now()
    history.record_answer(user_id, question, answer, now=now)
    # Same rule as build_context: no profile means no identified user, and
    # the archive's foreign key would reject the row anyway.
    if user_id is None or store.get_profile(user_id) is None:
        return
    try:
        store.archive_exchange(user_id, question, answer, now=now)
    except Exception:                                          # noqa: BLE001
        # The archive is a nice-to-have. A failure here must not surface to
        # the orchestrator and take the assistant's voice away mid-sentence.
        log.exception("could not archive exchange")


def _blocks_for(user_id: str, transcript: str,
                intent: str, now: datetime) -> list[str]:
    """Which data blocks belong in the prompt for this intent.

    Key decision: blocks are selective. A question about the weather should
    not drag the schedule into context. That is partly token thrift, but
    mostly because a small model grabs at whatever is in front of it and
    answers the wrong question.
    """
    blocks = []

    # "Повтори": hand back what was actually said last time. Recomputing the
    # answer risks saying something different from what the person half-heard.
    if intent == router.REPEAT:
        # `now` must be threaded through rather than letting history fall back
        # to the wall clock: otherwise a caller passing a fixed time gets its
        # own recorded answers rejected as stale.
        last = history.get_last_answer(user_id, now=now)
        return [prompts.REPEAT_BLOCK.format(answer=last) if last
                else prompts.REPEAT_EMPTY]

    # "Запомни, что...": the only branch that writes. The fact is stored here
    # so it is searchable immediately, before the model confirms it out loud.
    if intent == router.REMEMBER:
        fact = router.extract_fact(transcript)

        # A preference against something is stored twice over: as text, which
        # is what RAG recites, and as a subject, which is what places.py
        # filters on. Reciting a preference is not the same as honouring one,
        # and only the second column makes the assistant act on it.
        polarity = subject = None
        if router.is_dislike(transcript):
            polarity = store.NEGATIVE
            subject = llm_intent.extract_subject(fact)

        store.add_fact(user_id, fact, polarity=polarity, subject=subject)
        if subject:
            return [prompts.REMEMBER_DISLIKE_BLOCK.format(subject=subject)]
        return [prompts.REMEMBER_BLOCK.format(fact=fact)]

    if intent == router.SCHEDULE:
        today = now.date()
        day = router.resolve_day(transcript, today)
        label = router.day_label(day, today)
        events = store.get_schedule(user_id, day)
        if events:
            lines = "\n".join(f"- {time} {title}" for time, title in events)
            blocks.append(prompts.SCHEDULE_BLOCK.format(day=label, lines=lines))
        else:
            blocks.append(prompts.SCHEDULE_EMPTY.format(day=label))

    # Facts help where the answer depends on someone's preferences,
    # and are dead weight in a question about the weather.
    if intent in (router.PLACES, router.GENERAL):
        # On a places question, a dislike that has already been applied as a
        # filter is left out: quoting it would put the very name that was
        # just removed from the results back into the prompt.
        facts = store.get_facts(user_id, transcript,
                                skip_acted_on=(intent == router.PLACES))
        if facts:
            lines = "\n".join(f"- {fact}" for fact in facts)
            blocks.append(prompts.FACTS_BLOCK.format(lines=lines))

    # Past conversations only for open-ended questions. A schedule or weather
    # question gains nothing from old chatter, and a small model will happily
    # answer the old question instead of the new one if both are in view.
    # Turns still inside the live buffer are excluded: they are already in
    # the messages and would otherwise appear twice.
    if intent == router.GENERAL:
        past = store.get_dialogue(user_id, transcript, limit=2,
                                  before=now - history.TTL)
        if past:
            lines = "\n".join(
                prompts.DIALOGUE_LINE.format(question=_clip(q), answer=_clip(a))
                for q, a in past
            )
            blocks.append(prompts.DIALOGUE_BLOCK.format(lines=lines))

    # STEP 6: a weather block and a 2GIS block go here -- those will be HTTP
    # requests made at question time, not data read from the database.

    return blocks


def _clip(text: str, limit: int = 160) -> str:
    """Keep recalled turns short. Answers are meant to be one or two
    sentences, but a runaway one should not eat the context budget."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _format_now(now: datetime) -> str:
    """'среда, двадцать третье сентября, час сорок четыре минуты'."""
    return (f"{router.WEEKDAYS[now.weekday()]}, "
            f"{num_to_words.date_words(now.date())}, "
            f"{num_to_words.time_words(now.hour, now.minute)}")
