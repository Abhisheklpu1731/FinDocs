import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Loads chunks + their embeddings once, then answers many searches."""

    def __init__(self, strategy: str = "recursive"):
        self.model = SentenceTransformer(MODEL_NAME)
        with (DATA_DIR / f"chunks_{strategy}.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.vectors = np.load(DATA_DIR / "embeddings" / f"{strategy}_bge-small.npy")

    def search(self, question: str, k: int = 5) -> list[dict]:
        q = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        scores = self.vectors @ q                 # one score per chunk
        top = np.argsort(-scores)[:k]             # positions of the k highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    retriever = DenseRetriever()
    question = " ".join(sys.argv[1:]) or "What was Infosys revenue in FY25?"
    print(f"Q: {question}\n")
    for r in retriever.search(question):
        print(f"{r['score']:.3f}  [{r['doc_id']}, page {r['page']}]")
        print(f"       {r['text'][:200]!r}\n")