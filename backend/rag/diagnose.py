import json
import os
from collections import Counter

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "indexes", "rag_index")
)
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.30
MIN_CONTEXT_LENGTH = 100

QUERIES = [
    "What documents are required to open a savings account?",
    "What is the interest rate on a fixed deposit?",
    "How do I apply for a personal loan?",
    "What are the KYC requirements for new customers?",
    "How can I reset my internet banking password?",
]


def load_index() -> tuple[faiss.Index, list[str], list[dict]]:
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, chunks, meta


def format_source(meta: dict) -> str:
    source = meta.get("source", "doc")
    page = meta.get("page")
    row = meta.get("row")
    if page:
        return f"{source}, page {page}"
    if row:
        return f"{source}, row {row}"
    return source


def classify_failure(scores: list[float], chunks_passed: int, context_length: int) -> str:
    if TOP_K in {1, 2}:
        return "Top-k too low"
    if scores and scores[0] < SIMILARITY_THRESHOLD:
        return "Threshold too high"
    if chunks_passed == 0 and scores:
        return "Threshold too high"
    if context_length < MIN_CONTEXT_LENGTH:
        return "Empty context to LLM"
    if chunks_passed >= 3 and context_length >= MIN_CONTEXT_LENGTH:
        return "Retrieval healthy (LLM end)"
    return "Empty context to LLM"


def main() -> None:
    index, chunks, metadata = load_index()
    model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    summary = Counter()

    for query in QUERIES:
        q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, ids = index.search(q_vec, TOP_K)

        raw_scores = [float(score) for score in scores[0] if score > -1]
        context_blocks: list[str] = []
        chunks_passed = 0

        for score, idx_value in zip(scores[0], ids[0]):
            if idx_value < 0:
                continue
            if float(score) < SIMILARITY_THRESHOLD:
                continue
            meta = metadata[int(idx_value)]
            context_blocks.append(
                f"[Source: {format_source(meta)}]\n{chunks[int(idx_value)]}"
            )
            chunks_passed += 1

        context = "\n\n---\n\n".join(context_blocks)
        failure_point = classify_failure(raw_scores, chunks_passed, len(context))
        summary[failure_point] += 1

        padded_scores = raw_scores + [0.0] * (TOP_K - len(raw_scores))
        print(f"QUERY: {query}")
        print("-" * 41)
        print(f"Top-5 raw similarity scores : {[round(score, 4) for score in padded_scores[:TOP_K]]}")
        print(f"Current threshold            : {SIMILARITY_THRESHOLD:.2f}")
        print(f"Chunks passed threshold      : {chunks_passed} / {TOP_K}")
        print(f"Context length sent to LLM   : {len(context)} characters")
        print(f"Likely failure point         : {failure_point}")
        print("-" * 41)
        print()

    print("SUMMARY TABLE:")
    print("Failure point               | Queries affected")
    print("----------------------------|------------------")
    for label in [
        "Threshold too high",
        "Top-k too low",
        "Empty context to LLM",
        "Retrieval healthy (LLM end)",
    ]:
        print(f"{label:<28}| {summary[label]} / {len(QUERIES)}")


if __name__ == "__main__":
    main()
