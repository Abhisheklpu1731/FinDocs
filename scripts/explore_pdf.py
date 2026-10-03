import fitz  # PyMuPDF's import name (historical reasons)

doc = fitz.open("data/raw/lenskart_2025.pdf")
print("Total pages:", len(doc))

for page_num in [0, 5, 50]:          # look at 3 different pages
    page = doc[page_num]             # pages are 0-indexed in code
    text = page.get_text()
    print("\n" + "=" * 60)
    print(f"PDF page {page_num + 1} | printed label: {page.get_label()!r}")
    print(f"Characters: {len(text)}")
    print("=" * 60)
    print(text[:800])                # first 800 characters only