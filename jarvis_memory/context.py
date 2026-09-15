"""
Сборка контекста — ядро модуля Ц.

Здесь нет ни одного вызова языковой модели. Всё, что делает build_context, —
это несколько выборок данных и подстановка их в шаблон строками. Единственный
инференс LLM во всей цепочке происходит позже и снаружи: оркестратор берёт
готовые сообщения и отдаёт их Qwen ровно один раз.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from . import history, prompts, router, store

_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг",
             "пятница", "суббота", "воскресенье"]


@dataclass
class PromptContext:
    """Готовый контекст для одного запроса к LLM."""

    system_prompt: str
    history: list[dict[str, str]]
    transcript: str
    user_id: str | None
    intent: str
    debug: dict = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """Полный список сообщений — то, что оркестратор передаёт в модель."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": self.transcript},
        ]

    def estimate_tokens(self) -> int:
        """Грубая оценка бюджета: русский текст — примерно 2.5 символа на токен.

        Точность тут не нужна, нужен порядок величины: если оценка ползёт
        к тысяче, значит история и блоки данных вот-вот вытеснят друг друга.
        """
        chars = sum(len(m["content"]) for m in self.to_messages())
        return int(chars / 2.5)


def build_context(user_id: str | None, transcript: str,
                  now: datetime | None = None) -> PromptContext:
    """Собрать контекст по (кто спросил, что спросил).

    user_id = None означает, что модуль Б не опознал говорящего.
    """
    now = now or datetime.now()
    intent = router.route(transcript)
    profile = store.get_profile(user_id) if user_id else None

    # Профиль не нашёлся — значит, опознанного пользователя нет,
    # как бы ни выглядел переданный user_id.
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
    """Какие блоки данных класть в промпт для этого интента.

    Ключевое решение: блоки подставляются выборочно. Вопрос про погоду не
    должен тащить в контекст расписание — это не только экономия токенов,
    маленькая модель ещё и хватается за лишние данные и отвечает не на то.
    """
    blocks = []

    if intent == router.SCHEDULE:
        events = store.get_schedule(user_id, today)
        if events:
            lines = "\n".join(f"- {time} {title}" for time, title in events)
            blocks.append(prompts.SCHEDULE_BLOCK.format(lines=lines))
        else:
            blocks.append(prompts.SCHEDULE_EMPTY)

    # Факты полезны там, где ответ зависит от предпочтений человека,
    # и бесполезны в вопросе про погоду.
    if intent in (router.PLACES, router.GENERAL):
        facts = store.get_facts(user_id, transcript)
        if facts:
            lines = "\n".join(f"- {fact}" for fact in facts)
            blocks.append(prompts.FACTS_BLOCK.format(lines=lines))

    # ШАГ 6: сюда добавится блок погоды и блок 2GIS — это будут HTTP-запросы
    # в момент вопроса, а не данные из базы.

    return blocks


def _format_now(now: datetime) -> str:
    return f"{_WEEKDAYS[now.weekday()]}, {now.day:02d}.{now.month:02d}, {now:%H:%M}"
