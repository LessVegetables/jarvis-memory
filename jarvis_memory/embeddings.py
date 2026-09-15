"""
Turning text into vectors.

One job: take a sentence, return a list of floats, such that sentences with
similar meaning produce numerically close vectors. That closeness is the
whole trick behind RAG -- "можно мне миндальное печенье?" lands near
"аллергия на орехи" even though they share almost no letters.

There is no learning here, and nothing is generated. The model is a fixed
function, loaded once and called.

The model runs on the CPU via onnxruntime. It is NOT put on the NPU: that is
busy with speaker identification and the LLM, and a 29M-parameter model on
one short sentence does not need it.

Loading is lazy. Importing this module costs nothing until something is
actually embedded, so the orchestrator does not pay for RAG at startup if
nobody asks a question that needs it.

If the model file is missing, `is_available()` returns False and callers fall
back to a non-semantic query. That is deliberate: a teammate who has not
copied models/ across should still be able to run the whole module.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Callable, Sequence

log = logging.getLogger(__name__)

# rubert-tiny2 produces 312-dimensional vectors. If the model is swapped,
# this must change with it -- and the vec0 table has to be rebuilt, since
# its column width is fixed at creation.
DIM = 312

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_CANDIDATES = ("ru_embed.int8.onnx", "ru_embed.onnx")

# A callable(list[str]) -> list[list[float]], or None if not loaded yet.
_backend: Callable[[Sequence[str]], list[list[float]]] | None = None
_load_failed = False


def set_backend(fn: Callable[[Sequence[str]], list[list[float]]] | None) -> None:
    """Install an embedding function directly.

    Used by tests to inject a deterministic stand-in, and available if the
    team ever wants to swap the model without touching callers.
    """
    global _backend, _load_failed
    _backend = fn
    _load_failed = fn is None


def is_available() -> bool:
    """True if text can actually be embedded right now."""
    return _ensure_backend() is not None


def embed(texts: Sequence[str]) -> list[list[float]] | None:
    """Embed a batch of texts, or None if no embedder is available."""
    backend = _ensure_backend()
    if backend is None:
        return None
    if not texts:
        return []
    try:
        return backend(texts)
    except Exception:                                          # noqa: BLE001
        # A failure here must never take the assistant down mid-sentence:
        # the caller falls back to a non-semantic query and still answers.
        log.exception("embedding failed, falling back")
        return None


def serialize(vector: Sequence[float]) -> bytes:
    """Pack a vector the way sqlite-vec wants it: raw little-endian float32.

    The "<" matters: struct defaults to native byte order, which would write
    a different layout on a big-endian host and silently produce garbage
    distances. ARM is little-endian, but being explicit costs nothing.
    """
    return struct.pack(f"<{len(vector)}f", *vector)


def _ensure_backend():
    global _backend, _load_failed
    if _backend is not None or _load_failed:
        return _backend
    _backend = _load_onnx_backend()
    _load_failed = _backend is None
    return _backend


def _load_onnx_backend():
    """Build an ONNX-backed embedder, or return None with a one-line reason."""
    model_path = next(
        (_MODELS_DIR / name for name in _MODEL_CANDIDATES
         if (_MODELS_DIR / name).exists()),
        None,
    )
    if model_path is None:
        log.info("no embedding model in %s; semantic search disabled "
                 "(run tools/export_embedding_model.py on a laptop)", _MODELS_DIR)
        return None

    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        log.info("embedding dependencies missing (%s); semantic search disabled", exc)
        return None

    tokenizer_path = _MODELS_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        log.warning("model present but %s is missing", tokenizer_path)
        return None

    options = ort.SessionOptions()
    # One thread on purpose: the other cores are wanted by the LLM, STT and
    # the NPU pipeline. This model is small enough not to need them.
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options,
                                   providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding()
    input_names = {i.name for i in session.get_inputs()}
    log.info("embedding model loaded: %s", model_path.name)

    def encode(texts: Sequence[str]) -> list[list[float]]:
        encodings = tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = session.run(None, feed)[0]

        # Mean pooling over real tokens only. Including padding would drag
        # every vector towards zero by an amount that depends on how much
        # padding there happened to be -- i.e. on batch composition.
        expanded = mask[..., None].astype(hidden.dtype)
        pooled = (hidden * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1e-9)

        # L2-normalise so that Euclidean distance ranks identically to cosine
        # similarity. sqlite-vec's default metric is L2, and on unit vectors
        # the two orderings are the same, so this avoids configuring anything.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-9)).astype(float).tolist()

    return encode
