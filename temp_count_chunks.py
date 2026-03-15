import json
import os


INDEX_DIR = r"D:\Voice chatbot\indexes\rag_index"
chunks_path = os.path.join(INDEX_DIR, "chunks.json")

print(f"chunks_path={chunks_path}")
print(f"exists={os.path.exists(chunks_path)}")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"total_chunks={len(chunks)}")
