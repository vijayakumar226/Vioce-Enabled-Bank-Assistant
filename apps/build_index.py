import os
import glob
import json
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DOCS_DIR = "data/bank_docs"
INDEX_DIR = "indexes/rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_CHARS = 1200        # ~200-300 words
OVERLAP_CHARS = 200

def chunk_text(text: str, chunk_chars=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    text = " ".join(text.split())
    chunks = []
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

def load_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Turn each row into one text line for retrieval
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

    # Chunk everything and keep metadata
    chunks = []
    metas = []
    for text, meta in raw_items:
        for c in chunk_text(text):
            chunks.append(c)
            metas.append(meta)

    print(f"Loaded {len(raw_items)} doc parts -> created {len(chunks)} chunks")

    # Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(chunks, convert_to_numpy=True).astype("float32")

    # Cosine similarity index
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Save index + metadata + chunks
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print("✅ Saved index to rag_index/ (faiss.index, chunks.json, meta.json)")

if __name__ == "__main__":
    main()
