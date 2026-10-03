import pymupdf

PDF_PATH = "data/raw/lenskart_2025.pdf"
OCR_FONT_HINTS = ("OCR", "GlyphLess")   # font names OCR tools use

doc = pymupdf.open(PDF_PATH)
counts = {"digital": 0, "scanned": 0, "empty": 0}
scanned_pages = []
empty_pages = []

for i, page in enumerate(doc, start=1):
    text = page.get_text()

    fonts = set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fonts.add(span["font"])

    is_ocr = any(hint in font for font in fonts for hint in OCR_FONT_HINTS)

    if len(text.strip()) < 50:
        kind = "empty"
        empty_pages.append(i)
    elif is_ocr:
        kind = "scanned"
        scanned_pages.append(i)
    else:
        kind = "digital"
    counts[kind] += 1

print(counts)
print("Scanned pages:", scanned_pages)
print("Empty pages:", empty_pages)