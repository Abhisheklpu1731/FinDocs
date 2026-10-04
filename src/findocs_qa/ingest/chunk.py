import json
from pathlib import Path

PAGES_PATH = Path("data/processed/pages.jsonl")
OUT_DIR = Path("data/processed")
MIN_SECTION_WORDS = 40   # sections smaller than this get merged with the next
CHUNK_WORDS = 350    # roughly 500 tokens
OVERLAP_WORDS = 35   # roughly 50 tokens (fixed strategy only)
SEPARATORS = ["\n\n", "\n", ". ", " "]   # paragraph, line, sentence, word
MIN_CHUNK_WORDS = 8   # anything shorter is a footer, header or caption

def load_pages(path: Path) -> list[dict]:
    """Read pages.jsonl back into a list of dictionaries."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_chunk(page: dict, n: int, text: str) -> dict:
    """Build one chunk record that remembers where it came from."""
    return {
        "chunk_id": f"{page['doc_id']}_p{page['page']}_c{n}",
        "doc_id": page["doc_id"],
        "page": page["page"],
        "scanned": page["scanned"],
        "text": text,
    }


# ---------- Strategy 1: fixed-size ----------

def fixed_chunks(page: dict) -> list[dict]:
    """Cut one page into overlapping pieces of CHUNK_WORDS words."""
    words = page["text"].split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + CHUNK_WORDS]
        chunks.append(make_chunk(page, n, " ".join(piece)))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


# ---------- Strategy 2: recursive ----------

def split_recursive(text: str, max_words: int, separators: list[str]) -> list[str]:
    """Split text at the biggest natural break that keeps pieces small enough."""
    if len(text.split()) <= max_words:
        return [text]                      # already small enough: done

    sep, smaller_seps = separators[0], separators[1:]
    parts = text.split(sep)

    pieces = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate.split()) <= max_words:
            current = candidate            # still fits: keep adding
        else:
            if current:
                pieces.append(current)     # save what we have
            if len(part.split()) > max_words:
                # this one part is too big on its own: split it more finely
                pieces.extend(split_recursive(part, max_words, smaller_seps))
                current = ""
            else:
                current = part             # start a new piece with this part
    if current:
        pieces.append(current)
    return pieces


def recursive_chunks(page: dict) -> list[dict]:
    pieces = split_recursive(page["text"], CHUNK_WORDS, SEPARATORS)
    return [make_chunk(page, n, p.strip()) for n, p in enumerate(pieces) if p.strip()]

# ---------- Strategy 3: heading-based ----------

def is_heading(line: str) -> bool:
    """Guess whether a line is a section heading."""
    line = line.strip()
    words = line.split()
    if not (1 <= len(words) <= 8):
        return False                     # headings are short
    if line.endswith((".", ",", ":", ";")):
        return False                     # sentences end with punctuation
    if sum(ch.isdigit() for ch in line) > len(line) / 3:
        return False                     # mostly numbers = probably a table row
    return line.isupper() or line.istitle()


def heading_chunks(page: dict) -> list[dict]:
    """Cut a page into sections at each heading, keeping the heading on every piece."""
    # 1. Group lines into (heading, body) sections
    sections = []
    current_heading = ""
    current_lines = []
    for line in page["text"].split("\n"):
        if is_heading(line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
        # 1b. Merge tiny sections into the next one
    merged = []
    for heading, body in sections:
        if merged and len(merged[-1][1].split()) < MIN_SECTION_WORDS:
            prev_heading, prev_body = merged[-1]
            merged[-1] = (prev_heading, f"{prev_body}\n{heading}\n{body}".strip())
        else:
            merged.append((heading, body))
    # 2. Split long sections further, and put the heading on each piece
    chunks = []
    for heading, body in merged:
        for piece in split_recursive(body, CHUNK_WORDS, SEPARATORS):
            piece = piece.strip()
            if not piece:
                continue
            text = f"{heading}\n{piece}" if heading else piece
            chunks.append(make_chunk(page, len(chunks), text))
    return chunks

# ---------- Run all strategies ----------

STRATEGIES = {
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "heading": heading_chunks,
}


def main() -> None:
    pages = load_pages(PAGES_PATH)
    for name, chunk_fn in STRATEGIES.items():
        all_chunks = []
        for page in pages:
            all_chunks = [c for c in all_chunks if len(c["text"].split()) >= MIN_CHUNK_WORDS]
            all_chunks.extend(chunk_fn(page))

        out_path = OUT_DIR / f"chunks_{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        avg = sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks)
        print(f"{name:<10} {len(all_chunks):>5} chunks (avg {avg:.0f} words) -> {out_path}")


if __name__ == "__main__":
    main()