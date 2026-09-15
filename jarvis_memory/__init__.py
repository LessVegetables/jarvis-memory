"""
jarvis_memory -- Jarvis's memory and context module (task C).

What it does: turns (who asked, what they asked) into context for the LLM.
What it does not do: call a language model, or touch audio.

Use from the orchestrator (app.py):

    import jarvis_memory as memory

    ctx = memory.build_context(user_id, transcript)   # user_id=None if voice unrecognised
    answer = llm.generate(ctx.to_messages())
    memory.record_answer(user_id, transcript, answer)

The second call is required. Without it neither "repeat" nor multi-turn
dialogue works -- the module simply never learns what was said back.
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
