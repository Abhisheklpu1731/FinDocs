import json
from collections import Counter, defaultdict

pages_per_doc = Counter()
line_counts = defaultdict(Counter)

with open("data/processed/pages.jsonl", encoding="utf-8") as f:
    for row in f:
        rec = json.loads(row)
        pages_per_doc[rec["doc_id"]] += 1
        lines = {l.strip() for l in rec["text"].split("\n") if l.strip()}
        line_counts[rec["doc_id"]].update(lines)

for doc, total in pages_per_doc.items():
    print(f"\n{doc}: {total} pages (50% limit = {total / 2:.0f})")
    for line, n in line_counts[doc].most_common(5):
        print(f"   {n:>4} pages | {line[:70]!r}")