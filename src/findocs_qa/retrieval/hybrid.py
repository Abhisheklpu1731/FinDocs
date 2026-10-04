import sys

from findocs_qa.retrieval.bm25 import BM25Retriever
from findocs_qa.retrieval.dense import DenseRetriever

RRF_K = 60   # standard constant from the original RRF paper


class HybridRetriever:
    """Runs dense + BM25 search and merges their rankings with Reciprocal Rank Fusion."""

    def __init__(self, strategy: str = "recursive", dense=None, bm25=None):
        self.retrievers = {
            "dense": dense or DenseRetriever(strategy),
            "bm25": bm25 or BM25Retriever(strategy),
        }

    def search(
        self, question: str, k: int = 5, pool: int = 20, doc_ids: set[str] | None = None
    ) -> list[dict]:
        fused = {}
        for name, retriever in self.retrievers.items():
            for rank, chunk in enumerate(retriever.search(question, k=pool, doc_ids=doc_ids), start=1):
                cid = chunk["chunk_id"]
                if cid not in fused:
                    fused[cid] = {**chunk, "score": 0.0, "found_by": []}
                fused[cid]["score"] += 1 / (RRF_K + rank)
                fused[cid]["found_by"].append(f"{name}#{rank}")

        ranked = sorted(fused.values(), key=lambda c: c["score"], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    question = " ".join(sys.argv[1:]) or "What was Infosys revenue in FY25?"
    print(f"Q: {question}\n")
    for r in retriever.search(question):
        print(f"{r['score']:.4f}  [{r['doc_id']}, page {r['page']}]  found by: {r['found_by']}")
        print(f"       {r['text'][:200]!r}\n")