# backend/rag/rebuild.py
"""
Callable FAISS index builder.
Accepts PDF and TXT files from docs_dir, chunks + embeds them,
and writes faiss.index / chunks.json / meta.json to index_dir.
Returns the number of chunks indexed.
"""

import glob
import json
import os
import re

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 1200
OVERLAP_CHARS = 100
MIN_CHUNK_CHARS = 50

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    return _embed_model


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _chunk_paragraph(paragraph: str, chunk_chars: int, overlap: int) -> list[str]:
    sentences = _split_sentences(paragraph)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and len(candidate) > chunk_chars:
            chunk = " ".join(current).strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)

            carryover: list[str] = []
            carried_length = 0
            for existing in reversed(current):
                sentence_length = len(existing) + (1 if carryover else 0)
                if carryover and carried_length + sentence_length > overlap:
                    break
                carryover.insert(0, existing)
                carried_length += sentence_length

            current = carryover + [sentence]
        else:
            current.append(sentence)

        if len(sentence) > chunk_chars:
            if current and current[-1] == sentence and len(current) > 1:
                previous_chunk = " ".join(current[:-1]).strip()
                if len(previous_chunk) >= MIN_CHUNK_CHARS:
                    chunks.append(previous_chunk)
                current = [sentence]

            start = 0
            fragments: list[str] = []
            while start < len(sentence):
                end = min(len(sentence), start + chunk_chars)
                window = sentence[start:end]
                if end < len(sentence):
                    split_at = max(window.rfind(", "), window.rfind("; "), window.rfind(": "))
                    if split_at > MIN_CHUNK_CHARS:
                        end = start + split_at + 1
                        window = sentence[start:end]
                fragment = window.strip()
                if len(fragment) >= MIN_CHUNK_CHARS:
                    fragments.append(fragment)
                if end == len(sentence):
                    break
                start = max(end - overlap, start + 1)

            if fragments:
                chunks.extend(fragments[:-1])
                current = [fragments[-1]]

    final_chunk = " ".join(current).strip()
    if len(final_chunk) >= MIN_CHUNK_CHARS:
        chunks.append(final_chunk)

    return chunks


def _chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_chunk_paragraph(paragraph, chunk_chars, overlap))
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_CHARS]


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
        for chunk in _chunk_text(text):
            chunks.append(chunk)
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
