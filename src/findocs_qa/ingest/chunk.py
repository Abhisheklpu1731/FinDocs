import json
from pathlib import Path

PAGES_PATH = Path("data/processed/pages.jsonl")
OUT_PATH = Path("data/processed/chunks_fixed.jsonl")

CHUNK_WORDS = 350    # roughly 500 tokens
OVERLAP_WORDS = 35   # roughly 50 tokens


def load_pages(path: Path) -> list[dict]:
    """Read pages.jsonl back into a list of dictionaries."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fixed_chunks(page: dict) -> list[dict]:
    """Cut one page into overlapping chunks of CHUNK_WORDS words."""
    words = page["text"].split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + CHUNK_WORDS]
        chunks.append({
            "chunk_id": f"{page['doc_id']}_p{page['page']}_c{n}",
            "doc_id": page["doc_id"],
            "page": page["page"],
            "scanned": page["scanned"],
            "text": " ".join(piece),
        })
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def main() -> None:
    pages = load_pages(PAGES_PATH)
    all_chunks = []
    for page in pages:
        all_chunks.extend(fixed_chunks(page))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    avg = sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks)
    print(f"{len(pages)} pages -> {len(all_chunks)} chunks (avg {avg:.0f} words)")


if __name__ == "__main__":
    main()