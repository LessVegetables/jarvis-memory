"""
Короткая память диалога: последние реплики + последний ответ ("повтори").

Живёт в оперативной памяти, а не в базе: это состояние текущего разговора,
а не факт о пользователе. После перезапуска устройства его не жалко потерять.

Два ограничения, которых не было в телеграм-ботах на облачных моделях:

1. MAX_TURNS мал. Контекст локальной Qwen2.5-1.5B — это 2048 или 4096 токенов
   (число фиксируется при конвертации в .rkllm и потом не меняется). История
   конкурирует за этот бюджет с расписанием и фактами, то есть ровно с тем,
   ради чего проект и делается. Плюс маленькая модель тем хуже держит нить,
   чем длиннее контекст: на десятой реплике она начнёт отвечать на шестую.

2. TTL. Телеграм-чат — это непрерывный приватный тред: вернулся через неделю,
   и история всё ещё твоя и всё ещё уместна. Кухонная колонка — не тред.
   Если Антон спросил что-то в девять утра, а подошёл снова в семь вечера,
   подставлять утренние реплики нельзя: это уже другой разговор.
"""

from __future__ import annotations

from datetime import datetime, timedelta

MAX_TURNS = 3          # пар "вопрос-ответ", не сообщений
TTL = timedelta(minutes=5)

# user_id -> [(timestamp, вопрос, ответ), ...]
_buffers: dict[str, list[tuple[datetime, str, str]]] = {}

# История для неопознанного говорящего общая на всех — для прототипа сойдёт,
# но помнить об этом стоит: два разных гостя увидят реплики друг друга.
_UNKNOWN_KEY = "_unknown"


def _key(user_id: str | None) -> str:
    return user_id or _UNKNOWN_KEY


def record_answer(user_id: str | None, question: str, answer: str,
                  now: datetime | None = None) -> None:
    """Запомнить обмен репликами. Оркестратор вызывает это после ответа LLM."""
    now = now or datetime.now()
    buf = _buffers.setdefault(_key(user_id), [])
    buf.append((now, question, answer))
    del buf[:-MAX_TURNS]


def get_history(user_id: str | None,
                now: datetime | None = None) -> list[dict[str, str]]:
    """Последние реплики в формате сообщений чата, свежее TTL.

    Формат [{"role": ..., "content": ...}] выбран не случайно: Qwen2.5-Instruct
    обучена на диалоговой разметке ChatML, и реплики, переданные как отдельные
    сообщения user/assistant, она понимает заметно лучше, чем тот же текст,
    вклеенный в системный промпт абзацем "Ранее в диалоге: ...".
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
    """Текст последнего ответа — основа для интента "повтори"."""
    now = now or datetime.now()
    buf = _buffers.get(_key(user_id), [])
    if not buf:
        return None
    ts, _, answer = buf[-1]
    return answer if now - ts <= TTL else None


def clear(user_id: str | None = None) -> None:
    """Сбросить историю (одного пользователя или всю). Нужно в тестах."""
    if user_id is None:
        _buffers.clear()
    else:
        _buffers.pop(_key(user_id), None)
