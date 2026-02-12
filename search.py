import json
import numpy as np
from core.embedder import Embedder
from core.index import VectorIndex

INDEX_PATH = "storage/index.faiss"
META_PATH = "storage/meta.json"

embedder = Embedder()

with open(META_PATH, "r") as f:
    metadata = json.load(f)

dim = embedder.encode(["test"]).shape[1]
index = VectorIndex(dim)
index.load(INDEX_PATH)

def search(query, k=5):
    q_vec = embedder.encode([query])
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores, indices):
        results.append({
            "score": float(score),
            "source": metadata[idx]["source"],
            "text": metadata[idx]["text"][:300]
        })

    return results

if __name__ == "__main__":
    import sys
    query = sys.argv[1]
    results = search(query)

    for r in results:
        print("\n---")
        print("Score:", r["score"])
        print("Source:", r["source"])
        print("Snippet:", r["text"])
