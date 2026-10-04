import json
from pathlib import Path

from findocs_qa.retrieval.bm25 import BM25Retriever
from findocs_qa.retrieval.dense import DenseRetriever
from findocs_qa.retrieval.hybrid import HybridRetriever
from findocs_qa.retrieval.rerank import RerankRetriever

GOLDEN_PATH = Path("evals/golden.jsonl")
K = 5
STRATEGIES = ["fixed", "recursive", "heading"]
RETRIEVERS = {
    "dense": DenseRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "rerank": RerankRetriever,
}


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(results: list[dict], item: dict) -> int | None:
    """Return the rank (1-based) of the first correct page, or None if missed."""
    for rank, r in enumerate(results, start=1):
        if r["doc_id"] == item["doc_id"] and r["page"] in item["pages"]:
            return rank
    return None


def evaluate(retriever, questions: list[dict]) -> tuple[float, float]:
    hits = 0
    reciprocal_ranks = 0.0
    for item in questions:
        rank = first_hit_rank(retriever.search(item["question"], k=K), item)
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1 / rank
    n = len(questions)
    return hits / n, reciprocal_ranks / n


def main() -> None:
    golden = load_golden()
    answerable = [q for q in golden if q["type"] != "unanswerable"]
    print(f"{len(answerable)} answerable questions, k={K}\n")
    print(f"{'strategy':<10} {'retriever':<8} {'hit@5':>6} {'MRR':>6}")
    print("-" * 33)

    for strategy in STRATEGIES:
        for name, retriever_class in RETRIEVERS.items():
            retriever = retriever_class(strategy)
            hit_rate, mrr = evaluate(retriever, answerable)
            print(f"{strategy:<10} {name:<8} {hit_rate:>6.2f} {mrr:>6.2f}")


if __name__ == "__main__":
    main()