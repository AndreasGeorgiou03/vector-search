from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json

from core.embedder import Embedder
from core.index import VectorIndex
from core.reranker import Reranker

INDEX_PATH = "storage/index.faiss"
META_PATH = "storage/meta.json"

app = FastAPI(title="Vector Search API")

embedder = Embedder()
reranker = Reranker()  # loads cross-encoder once

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

dim = embedder.encode(["test"]).shape[1]
index = VectorIndex(dim)
index.load(INDEX_PATH)


class SearchRequest(BaseModel):
    query: str
    k: int = 5

    rerank: bool = False         # turn cross-encoder reranking on/off
    candidate_k: int = 20        # how many to retrieve before reranking

    source: Optional[str] = None
    min_score: Optional[float] = None


class SearchResult(BaseModel):
    chunk_id: int
    score: float
    source: str
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


@app.get("/")
def root():
    return {"message": "Vector Search API is running. Visit /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    # 1) retrieve candidates with FAISS
    q_vec = embedder.encode([req.query])

    # ensure candidate_k >= k and not more than dataset size
    candidate_k = max(req.candidate_k, req.k)
    candidate_k = min(candidate_k, len(metadata))

    vec_scores, vec_indices = index.search(q_vec, candidate_k)

    # build candidate list with optional vector-score filtering
    candidates = []
    for score, idx in zip(vec_scores, vec_indices):
        if idx == -1:
            continue

        item = metadata[idx]
        src = item["source"]

        if req.source and src != req.source:
            continue
        if req.min_score is not None and float(score) < req.min_score:
            continue

        candidates.append((int(idx), float(score), src, item["text"]))

    # If nothing survives filters
    if not candidates:
        return SearchResponse(query=req.query, results=[])

    # 2) optionally rerank with cross-encoder
    if req.rerank:
        # prepare (chunk_id, text) for reranker
        rerank_input = [(cid, text) for (cid, _, _, text) in candidates]
        reranked = reranker.rerank(req.query, rerank_input)

        # map rerank scores back to candidate data
        cand_by_id = {cid: (vec_score, src, text) for (cid, vec_score, src, text) in candidates}

        results = []
        for cid, rr_score in reranked[: req.k]:
            vec_score, src, text = cand_by_id[cid]
            results.append(SearchResult(
                chunk_id=cid,
                score=rr_score,      # now score = reranker score
                source=src,
                text=text
            ))
        return SearchResponse(query=req.query, results=results)

    # 3) no rerank: return top-k by vector score
    candidates.sort(key=lambda x: x[1], reverse=True)
    results = [
        SearchResult(chunk_id=cid, score=vec_score, source=src, text=text)
        for (cid, vec_score, src, text) in candidates[: req.k]
    ]
    return SearchResponse(query=req.query, results=results)
