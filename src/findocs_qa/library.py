import json
import re
import shutil
from pathlib import Path

import numpy as np
import pymupdf
from sentence_transformers import CrossEncoder, SentenceTransformer

from findocs_qa.retrieval.bm25 import BM25Retriever
from findocs_qa.retrieval.dense import MODEL_NAME, DenseRetriever
from findocs_qa.retrieval.hybrid import HybridRetriever
from findocs_qa.retrieval.rerank import RERANK_MODEL, RerankRetriever

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
UPLOAD_DIR = Path("data/uploads")
BUILTIN_LABELS = {
    "hdfc_bank_2025": "HDFC Bank",
    "infosys_2025": "Infosys",
    "lenskart_2025": "Lenskart",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "document"


class Library:
    """All searchable documents (built-in + uploaded) and the retriever over them.

    Models load once. After an upload or delete the indexes are rebuilt, which
    is fast: BM25 over a few thousand passages and a matrix of vectors.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.docs: dict[str, dict] = {}
        self._chunks: dict[str, list[dict]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._load_builtin()
        self._load_uploads()
        self._rebuild()

    # ---------- loading ----------

    def _load_builtin(self) -> None:
        chunks_path = DATA_DIR / "chunks_recursive.jsonl"
        vectors_path = DATA_DIR / "embeddings" / "recursive_bge-small.npy"
        if not (chunks_path.exists() and vectors_path.exists()):
            return
        with chunks_path.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        vectors = np.load(vectors_path)
        for doc_id in dict.fromkeys(c["doc_id"] for c in chunks):
            idx = [i for i, c in enumerate(chunks) if c["doc_id"] == doc_id]
            self._add(doc_id, BUILTIN_LABELS.get(doc_id, doc_id),
                      [chunks[i] for i in idx], vectors[idx],
                      RAW_DIR / f"{doc_id}.pdf", builtin=True)

    def _load_uploads(self) -> None:
        for meta_path in sorted(UPLOAD_DIR.glob("*/meta.json")):
            folder = meta_path.parent
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f]
            self._add(meta["doc_id"], meta["label"], chunks,
                      np.load(folder / "vectors.npy"), folder / "source.pdf",
                      builtin=False, stats=meta.get("stats"))

    def _add(self, doc_id, label, chunks, vectors, pdf_path, builtin, stats=None) -> None:
        self._chunks[doc_id] = chunks
        self._vectors[doc_id] = vectors
        self.docs[doc_id] = {
            "doc_id": doc_id,
            "label": label,
            "builtin": builtin,
            "pdf_path": pdf_path,
            "n_chunks": len(chunks),
            "n_pages": len({c["page"] for c in chunks}),
            "stats": stats,
        }

    def _rebuild(self) -> None:
        chunks = [c for d in self._chunks.values() for c in d]
        if not chunks:
            self.retriever = None
            return
        vectors = np.vstack(list(self._vectors.values()))
        hybrid = HybridRetriever(
            dense=DenseRetriever(chunks=chunks, vectors=vectors, model=self.embedder),
            bm25=BM25Retriever(chunks=chunks),
        )
        self.retriever = RerankRetriever(hybrid=hybrid, model=self.reranker)

    # ---------- changes ----------

    def unique_id(self, name: str) -> str:
        base = slugify(Path(name).stem)
        doc_id, n = base, 2
        while doc_id in self.docs:
            doc_id, n = f"{base}_{n}", n + 1
        return doc_id

    def add_upload(self, doc_id: str, label: str, pdf_bytes: bytes) -> Path:
        """Save the original PDF to disk and return its path (processing happens next)."""
        folder = UPLOAD_DIR / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "source.pdf"
        path.write_bytes(pdf_bytes)
        return path

    def commit_upload(self, doc_id: str, label: str, result: dict) -> None:
        """Persist a processed upload and make it searchable."""
        folder = UPLOAD_DIR / doc_id
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in result["chunks"]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(folder / "vectors.npy", result["vectors"])
        (folder / "meta.json").write_text(
            json.dumps({"doc_id": doc_id, "label": label, "stats": result["stats"]}),
            encoding="utf-8")
        self._add(doc_id, label, result["chunks"], result["vectors"],
                  folder / "source.pdf", builtin=False, stats=result["stats"])
        self._rebuild()

    def discard_upload(self, doc_id: str) -> None:
        """Remove an upload's files (used on failure and on delete)."""
        shutil.rmtree(UPLOAD_DIR / doc_id, ignore_errors=True)

    def delete(self, doc_id: str) -> None:
        if self.docs.get(doc_id, {}).get("builtin", True):
            return
        self.discard_upload(doc_id)
        for store in (self.docs, self._chunks, self._vectors):
            store.pop(doc_id, None)
        self._rebuild()

    # ---------- display helpers ----------

    def label(self, doc_id: str) -> str:
        return self.docs.get(doc_id, {}).get("label", doc_id)

    def page_image(self, doc_id: str, page: int, zoom: float = 1.4) -> bytes | None:
        """Render one PDF page as PNG so the user can see the source."""
        path = self.docs.get(doc_id, {}).get("pdf_path")
        if not path or not Path(path).exists():
            return None
        with pymupdf.open(path) as doc:
            return doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
