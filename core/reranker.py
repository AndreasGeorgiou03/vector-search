from sentence_transformers import CrossEncoder
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        # candidates: [(chunk_id, text), ...]
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            [(cid, float(s)) for (cid, _), s in zip(candidates, scores)],
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
