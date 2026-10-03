import pymupdf

doc = pymupdf.open("data/raw/lenskart_2025.pdf")

for page_num in [1, 6, 51]:
    page = doc[page_num - 1]
    fonts = set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fonts.add(span["font"])
    print(f"Page {page_num}: fonts = {fonts}")