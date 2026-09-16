#!/usr/bin/env python3
"""
Export the Russian embedding model to ONNX. RUN THIS ON A LAPTOP, NOT THE BOARD.

    pip install torch transformers onnx onnxscript onnxruntime    # laptop only!
    python3 tools/export_embedding_model.py

Writes models/ru_embed.onnx (+ tokenizer.json) and prints the file sizes.

Why laptop-only: torch and transformers are multi-gigabyte on ARM64 and would
eat a fifth of the board's 15 GB disk. The board never needs them -- at
runtime it loads the .onnx file with onnxruntime (21 MB) and tokenizes with
`tokenizers` (the small Rust package). Copy the models/ directory across.

Model: cointegrated/rubert-tiny2 -- ~29M parameters, 312-dim embeddings,
Russian-specific. Chosen over multilingual-e5-small (118M params, ~470 MB
fp32) purely because of the 4 GB RAM budget shared with a 1.7 GB LLM.
"""

import sys
from pathlib import Path

MODEL_NAME = "cointegrated/rubert-tiny2"
OUT_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print(__doc__)
        print("ERROR: torch/transformers missing. Install them on your LAPTOP.")
        print("       (torch >= 2.5 also needs onnxscript, even for the classic exporter)")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Downloading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

    # The tokenizer travels as a single tokenizer.json that the `tokenizers`
    # package can load on the board without transformers.
    tokenizer.save_pretrained(OUT_DIR)

    onnx_path = OUT_DIR / "ru_embed.onnx"
    sample = tokenizer("пример текста", return_tensors="pt")
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        # Both axes dynamic: sentences vary in length and we embed in batches
        # when seeding, one at a time at request time.
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )
    inputs = (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])

    # Both exporters call the wrapped module with a mix of positional and
    # keyword arguments. transformers >= 5 adds arguments to
    # BertModel.forward() that then collide with the positional ones
    # ("got multiple values for argument 'use_cache'"). The wrapper pins the
    # three inputs we actually export by name, so the export no longer
    # depends on the argument ORDER of somebody else's forward().
    class EmbedWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    wrapper = EmbedWrapper(model).eval()

    try:
        # torch >= 2.5 defaults to a new exporter that needs the onnxscript
        # package; dynamo=False selects the classic one, which does not.
        torch.onnx.export(wrapper, inputs, onnx_path, dynamo=False, **export_kwargs)
    except (TypeError, RuntimeError):
        # Older torch: no dynamo argument. Newer torch (>= 2.9): the classic
        # exporter is gone and dynamo=False falls back to the new one anyway,
        # which needs onnxscript installed.
        torch.onnx.export(wrapper, inputs, onnx_path, **export_kwargs)
    print(f"  wrote {onnx_path} ({onnx_path.stat().st_size / 1024**2:.1f} MB)")

    # int8 roughly quarters the file and the resident footprint. Accuracy loss
    # on short sentences is small; if retrieval looks worse, ship the fp32 file
    # instead -- the board can afford it if the LLM leaves room.
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = OUT_DIR / "ru_embed.int8.onnx"
        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
        print(f"  wrote {int8_path} ({int8_path.stat().st_size / 1024**2:.1f} MB)")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  int8 quantisation skipped: {exc}")

    print("\nCopy the models/ directory to the board. Do NOT install "
          "torch or transformers there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
