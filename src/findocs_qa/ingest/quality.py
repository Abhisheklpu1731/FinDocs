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