from pathlib import Path

import numpy as np

from findocs_qa.ingest.chunk import MIN_CHUNK_WORDS, recursive_chunks
from findocs_qa.ingest.parse import iter_pages

EMBED_BATCH = 32
STAGES = ["parse", "chunk", "embed", "index"]


def ingest_pdf(pdf_path: Path, doc_id: str, model):
    """Process one PDF end to end, yielding progress events as it goes.

    Every event is a dict: {stage, done, total, message}. The last event has
    stage == "done" and carries the finished result under "result":
    {"pages": [...], "chunks": [...], "vectors": ndarray, "stats": {...}}.
    """
    # 1. parse: one page at a time so progress is real
    pages, empty, n_pages = [], 0, 0
    for i, total, rec in iter_pages(pdf_path, doc_id):
        n_pages = total
        if rec is None:
            empty += 1
        else:
            pages.append(rec)
        yield {"stage": "parse", "done": i, "total": total,
               "message": f"Reading page {i} of {total}"}
    if not pages:
        raise ValueError("No readable text found. The PDF may be fully scanned or image-only.")

    # 2. chunk: split every page into passages
    chunks = []
    for n, page in enumerate(pages, start=1):
        chunks.extend(c for c in recursive_chunks(page)
                      if len(c["text"].split()) >= MIN_CHUNK_WORDS)
        yield {"stage": "chunk", "done": n, "total": len(pages),
               "message": f"Splitting page {page['page']} into passages ({len(chunks)} so far)"}
    if not chunks:
        raise ValueError("The PDF had text but no usable passages after cleaning.")

    # 3. embed: turn passages into vectors, one batch at a time
    texts = [c["text"] for c in chunks]
    batches = []
    for start in range(0, len(texts), EMBED_BATCH):
        batches.append(model.encode(texts[start:start + EMBED_BATCH],
                                    batch_size=EMBED_BATCH, normalize_embeddings=True))
        done = min(start + EMBED_BATCH, len(texts))
        yield {"stage": "embed", "done": done, "total": len(texts),
               "message": f"Embedding passages {done} of {len(texts)}"}
    vectors = np.vstack(batches)

    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    stats = {
        "total_pages": n_pages,
        "text_pages": len(pages),
        "empty_pages": empty,
        "scanned_pages": sum(p["scanned"] for p in pages),
        "chunks": len(chunks),
        "avg_words": round(avg_words),
    }
    yield {"stage": "index", "done": 0, "total": 1, "message": "Building the search index"}
    yield {"stage": "done", "done": 1, "total": 1, "message": "Ready",
           "result": {"pages": pages, "chunks": chunks, "vectors": vectors, "stats": stats}}
