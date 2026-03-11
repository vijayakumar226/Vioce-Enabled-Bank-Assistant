# backend/rag/pipeline.py
import os
import json
import threading

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "indexes", "rag_index")
)
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TOP_K = 3
SIM_THRESHOLD = 0.45

# ── Thread-safe index cache ───────────────────────────────────────────────────
# We store all mutable state inside a dict so reload_index() can swap
# everything atomically under a lock without re-assigning module globals
# (which would not be visible to other threads holding a reference).

_lock = threading.Lock()
_cache: dict = {
    "index": None,
    "chunks": [],
    "meta": [],
}


def _load_index_from_disk() -> dict:
    idx = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {"index": idx, "chunks": chunks, "meta": meta}


def reload_index() -> None:
    """Hot-swap the in-memory FAISS index from disk. Thread-safe."""
    new = _load_index_from_disk()
    with _lock:
        _cache["index"] = new["index"]
        _cache["chunks"] = new["chunks"]
        _cache["meta"] = new["meta"]


# Load on first import
reload_index()

# ── Embedder & LLM client (loaded once) ──────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found. Set HF_TOKEN environment variable and restart."
    )
client = InferenceClient(token=hf_token)


# ── Core functions ────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[tuple[float, str, dict]]:
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    with _lock:
        idx = _cache["index"]
        chunks = _cache["chunks"]
        meta = _cache["meta"]

    scores, ids = idx.search(q_vec, TOP_K)
    results: list[tuple[float, str, dict]] = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        results.append((float(s), chunks[int(i)], meta[int(i)]))
    return results


def answer_query(query: str, attachment_contexts: list[dict] | None = None) -> dict:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please type a question.", "sources": []}

    hits = retrieve(query)

    context_lines: list[str] = []
    sources: list[dict] = []

    if hits and hits[0][0] >= SIM_THRESHOLD:
        for score, text, m in hits:
            src = m.get("source", "doc")
            page = m.get("page")
            row = m.get("row")
            loc = f"page {page}" if page else (f"row {row}" if row else "")
            context_lines.append(f"[{src} {loc}] {text}")
            sources.append(
                {
                    "score": score,
                    "text": text,
                    "source": src,
                    "page": page,
                    "row": row,
                    "kind": "rag",
                }
            )

    attachment_contexts = attachment_contexts or []
    for attachment in attachment_contexts:
        name = attachment.get("name", "attachment")
        content = (attachment.get("content") or "").strip()
        if not content:
            continue

        context_lines.append(f"[attachment {name}] {content}")
        sources.append(
            {
                "score": 1.0,
                "text": content,
                "source": name,
                "page": None,
                "row": None,
                "kind": "attachment",
            }
        )

    if not context_lines:
        return {"answer": "I don't know from the given data.", "sources": []}

    context = "\n".join(context_lines)

    prompt = f"""You are a banking customer-support assistant.
Answer ONLY using the context. If not found, say: "I don't know from the given data."

Context:
{context}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Use only context. Do not guess."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    answer = resp.choices[0].message["content"].strip()
    return {"answer": answer, "sources": sources}
