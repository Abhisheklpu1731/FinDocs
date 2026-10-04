import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/processed")
OUT_DIR = Path("data/processed/embeddings")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
STRATEGIES = ["fixed", "recursive", "heading"]


def load_chunks(path: Path) -> list[dict]:
    """Read a chunks .jsonl file into a list of dictionaries."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    for strategy in STRATEGIES:
        chunks = load_chunks(CHUNKS_DIR / f"chunks_{strategy}.jsonl")
        texts = [c["text"] for c in chunks]

        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        out_path = OUT_DIR / f"{strategy}_bge-small.npy"
        np.save(out_path, vectors)
        print(f"{strategy:<10} {vectors.shape} -> {out_path}")


if __name__ == "__main__":
    main()