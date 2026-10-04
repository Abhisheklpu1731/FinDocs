import sys

from sentence_transformers import CrossEncoder

from findocs_qa.retrieval.hybrid import HybridRetriever

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRetriever:
    """Hybrid search for a wide shortlist, then a cross-encoder picks the best."""

    def __init__(self, strategy: str = "recursive", hybrid=None, model=None):
        self.hybrid = hybrid or HybridRetriever(strategy)
        self.model = model or CrossEncoder(RERANK_MODEL)

    def search(
        self,
        question: str,
        k: int = 5,
        pool: int = 20,
        doc_ids: set[str] | None = None,
        on_step=None,
    ) -> list[dict]:
        """on_step(name, detail) is called after each stage so a UI can show progress."""
        candidates = self.hybrid.search(question, k=pool, doc_ids=doc_ids)
        if on_step:
            on_step("hybrid", f"{len(candidates)} candidate passages (keyword + semantic)")

        pairs = [(question, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for hybrid_rank, (c, s) in enumerate(zip(candidates, scores), start=1):
            c["hybrid_rank"] = hybrid_rank
            c["score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        if on_step:
            on_step("rerank", f"kept the best {min(k, len(ranked))}")
        return ranked[:k]


if __name__ == "__main__":
    retriever = RerankRetriever()
    question = " ".join(sys.argv[1:]) or "What was Infosys revenue in FY25?"
    print(f"Q: {question}\n")
    for r in retriever.search(question):
        print(f"{r['score']:6.2f}  [{r['doc_id']}, page {r['page']}]  "
              f"was hybrid #{r['hybrid_rank']}, found by {r['found_by']}")
        print(f"        {r['text'][:200]!r}\n")