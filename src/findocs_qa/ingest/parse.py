import json
import re
from pathlib import Path
import unicodedata

import pymupdf

from findocs_qa.ingest.quality import page_kind

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/pages.jsonl")


def clean(text: str) -> str:
    """Tidy whitespace without changing any words or numbers."""
    text = unicodedata.normalize("NFKC", text)   # ﬁ -> fi, ﬂ -> fl, etc.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)          # many spaces -> one space
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # many blank lines -> one blank line
    return text.strip()


def iter_pages(pdf_path: Path, doc_id: str | None = None):
    """Yield (page_number, total_pages, record) for every page; record is None for empty pages."""
    doc_id = doc_id or pdf_path.stem
    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            kind = page_kind(page)
            if kind == "empty":
                yield i, total, None
                continue
            text = clean(page.get_text("text", sort=True))
            yield i, total, {
                "doc_id": doc_id,
                "page": i,
                "scanned": kind == "scanned",
                "text": text,
                "n_chars": len(text),
            }


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Read one PDF and return one record per non-empty page."""
    return [rec for _, _, rec in iter_pages(pdf_path) if rec is not None]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            records = parse_pdf(pdf_path)
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"{pdf_path.stem:<20} {len(records)} pages written")
    print(f"Done: {total} pages -> {OUT_PATH}")


if __name__ == "__main__":
    main()