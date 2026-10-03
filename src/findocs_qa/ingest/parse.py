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


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Read one PDF and return one record per non-empty page."""
    records = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            kind = page_kind(page)
            if kind == "empty":
                continue
            text = clean(page.get_text("text", sort=True))
            records.append({
                "doc_id": pdf_path.stem,
                "page": i,
                "scanned": kind == "scanned",
                "text": text,
                "n_chars": len(text),
            })
    return records


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