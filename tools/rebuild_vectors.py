#!/usr/bin/env python3
"""
Drop and recompute every stored vector. Run after changing the embedding:
a different model, a different pooling, or a different DIM.

    python3 tools/rebuild_vectors.py

Vectors from two different embeddings are not comparable. If the model
changes and the old vectors stay, search silently returns nonsense, which is
far harder to notice than an error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import db, embeddings, store  # noqa: E402


def main() -> int:
    if not db.vec_available():
        print("sqlite-vec is not available; nothing to rebuild.")
        return 1
    if not embeddings.is_available():
        print("No embedding model found; vectors cleared but not rebuilt.")
    counts = store.rebuild_vectors()
    print(f"Rebuilt vectors in {db.db_path()}")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
