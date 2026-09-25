import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config import RERANK_MODEL, TOP_K_RERANK, TOP_K_DENSE, RERANK_THRESHOLD
from retrieval.hybrid_retriever import hybrid_retrieve

# Lazy-loaded CrossEncoder reranker
_reranker = None

def get_reranker():
    """Lazy load CrossEncoder reranking model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading CrossEncoder reranker: {RERANK_MODEL}...")
        _reranker = CrossEncoder(RERANK_MODEL)
        print("Reranker loaded successfully.")
    return _reranker

def sigmoid(x: float) -> float:
    """Map real-valued cross-encoder logit into [0, 1] probability confidence score."""
    return float(1.0 / (1.0 + np.exp(-x)))

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = TOP_K_RERANK
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Rerank candidate chunks using CrossEncoder.
    Returns: (top_n_candidates, is_insufficient)
    """
    if not candidates:
        return [], True

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    raw_scores = reranker.predict(pairs)

    scored_candidates = []
    for i, c in enumerate(candidates):
        raw = float(raw_scores[i])
        norm_score = sigmoid(raw)  # Normalized 0 to 1 confidence
        item = dict(c)
        item["rerank_raw_score"] = raw
        item["rerank_score"] = norm_score
        scored_candidates.append(item)

    # Sort descending by rerank score
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    max_score = scored_candidates[0]["rerank_score"] if scored_candidates else 0.0
    is_insufficient = (max_score < RERANK_THRESHOLD)

    top_candidates = scored_candidates[:top_n]
    return top_candidates, is_insufficient

def retrieve_and_rerank(
    query: str,
    where_filter: Optional[Dict[str, Any]] = None,
    top_k_candidates: int = TOP_K_DENSE,
    top_k_final: int = TOP_K_RERANK
) -> Dict[str, Any]:
    """
    Complete retrieval pipeline:
    1. Hybrid retrieval (dense bge-small + sparse BM25 fused via RRF)
    2. Deep cross-encoder reranking
    3. Low-confidence threshold check
    """
    candidates = hybrid_retrieve(query, n=top_k_candidates, where_filter=where_filter)
    if not candidates:
        return {
            "chunks": [],
            "insufficient": True,
            "max_score": 0.0
        }

    top_chunks, insufficient = rerank(query, candidates, top_n=top_k_final)
    max_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0

    return {
        "chunks": top_chunks,
        "insufficient": insufficient,
        "max_score": max_score
    }
