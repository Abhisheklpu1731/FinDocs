import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from findocs_qa.retrieval.bm25 import doc_mask, top_k

DATA_DIR = Path("data/processed")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Loads chunks + their embeddings once, then answers many searches."""

    def __init__(
        self,
        strategy: str = "recursive",
        chunks: list[dict] | None = None,
        vectors: np.ndarray | None = None,
        model: SentenceTransformer | None = None,
    ):
        self.model = model or SentenceTransformer(MODEL_NAME)
        if chunks is None:
            with (DATA_DIR / f"chunks_{strategy}.jsonl").open(encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f]
            vectors = np.load(DATA_DIR / "embeddings" / f"{strategy}_bge-small.npy")
        self.chunks = chunks
        self.vectors = vectors

    def search(self, question: str, k: int = 5, doc_ids: set[str] | None = None) -> list[dict]:
        q = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        scores = self.vectors @ q                 # one score per chunk
        top = top_k(scores, k, doc_mask(self.chunks, doc_ids))
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    retriever = DenseRetriever()
    question = " ".join(sys.argv[1:]) or "What was Infosys revenue in FY25?"
    print(f"Q: {question}\n")
    for r in retriever.search(question):
        print(f"{r['score']:.3f}  [{r['doc_id']}, page {r['page']}]")
        print(f"       {r['text'][:200]!r}\n")
