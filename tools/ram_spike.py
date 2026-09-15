#!/usr/bin/env python3
"""
Step 0: prove the embedding model fits in the RAM budget. RUN ON THE BOARD.

    pip install onnxruntime tokenizers numpy
    python3 tools/ram_spike.py

Measures resident memory before and after loading the model, and how long one
embedding takes. Run it while the LLM is also loaded, since that is the
situation that actually matters.

The budget: 4 GB total, of which Qwen2.5-1.5B w8a8 takes ~1.7 GB, the OS
~350 MB, and STT/TTS/wake word ~400 MB. That leaves roughly 250-400 MB for
this module. If the number below lands above ~200 MB, say so early -- the
fallback is keyword retrieval with no model at all, and it is far better to
know that now than on the day before the defense.
"""

import resource
import sys
import time
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


def rss_mb() -> float:
    """Resident set size in MB. ru_maxrss is KB on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / 1024**2


def main() -> int:
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(f"ERROR: {exc}\nInstall: pip install onnxruntime tokenizers numpy")
        return 1

    model_path = next((MODELS / name for name in
                       ("ru_embed.int8.onnx", "ru_embed.onnx")
                       if (MODELS / name).exists()), None)
    if model_path is None:
        print(f"ERROR: no model in {MODELS}. Run tools/export_embedding_model.py "
              "on a laptop first, then copy models/ across.")
        return 1

    baseline = rss_mb()
    print(f"RSS before loading:   {baseline:7.1f} MB")
    print(f"Model file:           {model_path.stat().st_size / 1024**2:7.1f} MB"
          f"  ({model_path.name})")

    # Single-threaded on purpose: the NPU and the other models want the cores,
    # and a 29M-parameter model on one short sentence does not need four.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options,
                                   providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
    print(f"RSS after loading:    {rss_mb():7.1f} MB "
          f"(+{rss_mb() - baseline:.1f} MB)")

    sentences = [
        "у меня аллергия на орехи",
        "во сколько у меня сегодня лекция по матанализу",
        "я вегетарианец и не ем мясо",
    ]
    inputs = {i.name for i in session.get_inputs()}
    timings = []
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])

        started = time.perf_counter()
        hidden = session.run(None, feed)[0]
        timings.append((time.perf_counter() - started) * 1000)

        # Mean pooling over real tokens only -- padding must not drag the
        # vector towards zero.
        mask = feed["attention_mask"][..., None]
        vector = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if sentence is sentences[0]:
            print(f"Embedding dimension:  {vector.shape[-1]:7d}")

    print(f"RSS after inference:  {rss_mb():7.1f} MB")
    print(f"Latency per sentence: {sum(timings) / len(timings):7.1f} ms "
          f"(min {min(timings):.1f}, max {max(timings):.1f})")

    total = rss_mb() - baseline
    print()
    if total < 200:
        print(f"VERDICT: fits. ~{total:.0f} MB for the module, budget is 250-400 MB.")
    else:
        print(f"VERDICT: {total:.0f} MB is above the ~200 MB target. Try the int8 "
              "file if you used fp32, otherwise fall back to keyword retrieval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
