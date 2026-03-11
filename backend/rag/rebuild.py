# backend/rag/rebuild.py
"""
Callable FAISS index builder.
Accepts PDF and TXT files from docs_dir, chunks + embeds them,
and writes faiss.index / chunks.json / meta.json to index_dir.
Returns the number of chunks indexed.
"""

import os
import json
import glob

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

# Singleton embed model (reuse if already loaded elsewhere)
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> list[tuple[str, dict]]:
    reader = PdfReader(path)
    out: list[tuple[str, dict]] = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            out.append((t, {"source": os.path.basename(path), "page": i + 1}))
    return out


def _load_txt(path: str) -> list[tuple[str, dict]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
    if text.strip():
        return [(text, {"source": os.path.basename(path), "page": None})]
    return []


def _load_all_docs(docs_dir: str) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    for p in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        items.extend(_load_pdf(p))
    for p in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        items.extend(_load_txt(p))
    return items


# ── Public API ─────────────────────────────────────────────────────────────────

def rebuild_index(docs_dir: str, index_dir: str) -> int:
    """
    Build (or rebuild) the FAISS index from all PDF/TXT files in docs_dir.
    Writes faiss.index, chunks.json, meta.json to index_dir.
    Returns the total number of chunks indexed.
    Raises RuntimeError if no documents are found.
    """
    os.makedirs(index_dir, exist_ok=True)

    raw_items = _load_all_docs(docs_dir)
    if not raw_items:
        raise RuntimeError(f"No PDF/TXT documents found in {docs_dir!r}.")

    chunks: list[str] = []
    metas: list[dict] = []
    for text, meta in raw_items:
        for c in _chunk_text(text):
            chunks.append(c)
            metas.append(meta)

    model = _get_embed_model()
    vecs = model.encode(chunks, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)

    faiss.write_index(idx, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    return len(chunks)
