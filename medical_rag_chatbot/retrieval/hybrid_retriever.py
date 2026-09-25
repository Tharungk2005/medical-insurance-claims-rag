from typing import List, Dict, Any, Optional
from config import RRF_K, TOP_K_DENSE
from retrieval.dense_retriever import dense_retrieve
from retrieval.sparse_retriever import bm25_retrieve

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = RRF_K
) -> List[Dict[str, Any]]:
    """
    Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    # Helper key generator for deduplication
    def make_key(item: Dict[str, Any]) -> str:
        cid = item.get("metadata", {}).get("chunk_id")
        if cid:
            return cid
        return item.get("text", "")[:100]

    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        key = make_key(item)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_lookup:
            doc_lookup[key] = dict(item)
            doc_lookup[key]["dense_rank"] = rank
            doc_lookup[key]["dense_score"] = item.get("score", 0.0)

    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        key = make_key(item)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_lookup:
            doc_lookup[key] = dict(item)
        doc_lookup[key]["sparse_rank"] = rank
        doc_lookup[key]["sparse_score"] = item.get("score", 0.0)

    # Sort merged candidates by descending RRF score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for rank, key in enumerate(sorted_keys, start=1):
        candidate = doc_lookup[key]
        candidate["rrf_score"] = scores[key]
        candidate["fused_rank"] = rank
        fused_results.append(candidate)

    return fused_results

def hybrid_retrieve(
    query: str,
    n: int = TOP_K_DENSE,
    where_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Run dense (semantic) and sparse (BM25) retrievals, then fuse results using RRF.
    Returns top-N fused candidates.
    """
    dense_candidates = dense_retrieve(query, n=n, where_filter=where_filter)
    sparse_candidates = bm25_retrieve(query, n=n)

    # If metadata filtering is applied to dense, filter sparse results to match if possible
    if where_filter:
        filtered_sparse = []
        for s in sparse_candidates:
            meta = s.get("metadata", {})
            match = True
            for fk, fv in where_filter.items():
                if fk in meta and meta[fk] != fv:
                    match = False
                    break
            if match:
                filtered_sparse.append(s)
        sparse_candidates = filtered_sparse

    fused = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=RRF_K)
    return fused[:n]
