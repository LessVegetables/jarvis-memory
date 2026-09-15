"""
jarvis_memory — модуль памяти и контекста Джарвиса (задача Ц).

Что делает: по паре (кто спросил, что спросил) собирает контекст для LLM.
Чего не делает: не обращается к языковой модели, не работает со звуком.

Использование из оркестратора (app.py):

    import jarvis_memory as memory

    ctx = memory.build_context(user_id, transcript)   # user_id=None, если голос не опознан
    answer = llm.generate(ctx.to_messages())
    memory.record_answer(user_id, transcript, answer)

Второй вызов обязателен: без него не работают ни "повтори", ни диалог
из нескольких реплик — модуль просто не узнает, что именно было сказано.
"""

from .context import PromptContext, build_context
from .history import clear, get_last_answer, record_answer

__all__ = [
    "build_context",
    "PromptContext",
    "record_answer",
    "get_last_answer",
    "clear",
]
