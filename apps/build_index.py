import glob
import json
import os
import re

import faiss
import pandas as pd
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = "data/bank_docs"
INDEX_DIR = "indexes/rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_CHARS = 1200
OVERLAP_CHARS = 100
MIN_CHUNK_CHARS = 50


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


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_chunk_paragraph(paragraph, chunk_chars, overlap))
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_CHARS]


def load_pdf(path):
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            out.append((t, {"source": os.path.basename(path), "page": i + 1}))
    return out


def load_docx(path):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if text.strip():
        return [(text, {"source": os.path.basename(path), "page": None})]
    return []


def load_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
    if text.strip():
        return [(text, {"source": os.path.basename(path), "page": None})]
    return []


def load_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = []
    for idx, row in df.iterrows():
        line = " | ".join([f"{col}: {row[col]}" for col in df.columns if row[col]])
        if line.strip():
            out.append((line, {"source": os.path.basename(path), "row": int(idx) + 1}))
    return out


def load_xlsx(path):
    df = pd.read_excel(path, dtype=str, keep_default_na=False)
    out = []
    for idx, row in df.iterrows():
        line = " | ".join([f"{col}: {row[col]}" for col in df.columns if row[col]])
        if line.strip():
            out.append((line, {"source": os.path.basename(path), "row": int(idx) + 1}))
    return out


def load_all_docs(folder):
    items = []

    for p in glob.glob(os.path.join(folder, "*.pdf")):
        items.extend(load_pdf(p))

    for p in glob.glob(os.path.join(folder, "*.docx")):
        items.extend(load_docx(p))

    for p in glob.glob(os.path.join(folder, "*.txt")):
        items.extend(load_txt(p))

    for p in glob.glob(os.path.join(folder, "*.csv")):
        items.extend(load_csv(p))

    for p in glob.glob(os.path.join(folder, "*.xlsx")):
        items.extend(load_xlsx(p))

    return items


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    raw_items = load_all_docs(DOCS_DIR)
    if not raw_items:
        raise RuntimeError(f"No documents found in {DOCS_DIR}. Add PDFs/CSV/DOCX/XLSX.")

    chunks = []
    metas = []
    for text, meta in raw_items:
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metas.append(meta)

    print(f"Loaded {len(raw_items)} doc parts -> created {len(chunks)} chunks")

    model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    vecs = model.encode(chunks, convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print("Saved index to rag_index/ (faiss.index, chunks.json, meta.json)")


if __name__ == "__main__":
    main()
