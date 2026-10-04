
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OCR_FONT_HINTS = ("OCR", "GlyphLess")


def page_kind(page) -> str:
    """Classify one page as 'empty', 'scanned' or 'digital'."""
    text = page.get_text()
    if len(text.strip()) < 50:
        return "empty"

    fonts = set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fonts.add(span["font"])

    if any(hint in font for font in fonts for hint in OCR_FONT_HINTS):
        return "scanned"
    return "digital"


for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
    doc = pymupdf.open(pdf_path)
    counts = {"digital": 0, "scanned": 0, "empty": 0}
    for page in doc:
        counts[page_kind(page)] += 1
    scanned_pct = 100 * counts["scanned"] / len(doc)
    print(f"{pdf_path.stem:<20} pages={len(doc):<4} {counts}  scanned={scanned_pct:.0f}%")
    doc.close()
