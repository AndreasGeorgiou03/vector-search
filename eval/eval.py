import json
from typing import List, Dict

from core.embedder import Embedder
from core.index import VectorIndex
from core.reranker import Reranker

INDEX_PATH = "storage/index.faiss"
META_PATH = "storage/meta.json"
QUERIES_PATH = "eval/queries.json"

def load_meta() -> List[Dict]:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_queries() -> List[Dict]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_at_k(ranked_sources: List[str], relevant: set, k: int) -> float:
    return 1.0 if any(src in relevant for src in ranked_sources[:k]) else 0.0

def mrr(ranked_sources: List[str], relevant: set) -> float:
    for i, src in enumerate(ranked_sources, start=1):
        if src in relevant:
            return 1.0 / i
    return 0.0

def evaluate(use_rerank=False, candidate_k=10):
    meta = load_meta()
    queries = load_queries()

    embedder = Embedder()
    dim = embedder.encode(["test"]).shape[1]
    index = VectorIndex(dim)
    index.load(INDEX_PATH)

    reranker = Reranker() if use_rerank else None

    ks = [1, 3, 5]
    recall_scores = {k: 0.0 for k in ks}
    mrr_scores = 0.0

    for q in queries:
        query = q["query"]
        relevant = set(q["relevant_sources"])

        q_vec = embedder.encode([query])
        scores, indices = index.search(q_vec, candidate_k)

        candidates = [(idx, meta[idx]["text"]) for idx in indices if idx != -1]

        if use_rerank:
            reranked = reranker.rerank(query, candidates)
            ranked_sources = [meta[cid]["source"] for cid, _ in reranked]
        else:
            ranked_sources = [meta[idx]["source"] for idx, _ in candidates]

        for k in ks:
            recall_scores[k] += recall_at_k(ranked_sources, relevant, k)
        mrr_scores += mrr(ranked_sources, relevant)

    n = len(queries)
    print("=== RERANK ===" if use_rerank else "=== VECTOR ONLY ===")
    for k in ks:
        print(f"Recall@{k}: {recall_scores[k]/n:.3f}")
    print(f"MRR: {mrr_scores/n:.3f}")
    print()

if __name__ == "__main__":
    evaluate(use_rerank=False)
    evaluate(use_rerank=True)

