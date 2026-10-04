import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data/processed")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words and numbers.
    'Net Profit (FY25)' -> ['net', 'profit', 'fy25']
    """
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Builds a keyword index over the chunks once, then answers many searches."""

    def __init__(self, strategy: str = "recursive"):
        with (DATA_DIR / f"chunks_{strategy}.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, question: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(question))   # one score per chunk
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    retriever = BM25Retriever()
    question = " ".join(sys.argv[1:]) or "What was Infosys revenue in FY25?"
    print(f"Q: {question}\n")
    for r in retriever.search(question):
        print(f"{r['score']:.2f}  [{r['doc_id']}, page {r['page']}]")
        print(f"       {r['text'][:200]!r}\n")