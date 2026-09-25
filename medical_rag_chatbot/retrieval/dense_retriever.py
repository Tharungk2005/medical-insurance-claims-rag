from typing import List, Dict, Any, Optional
from ingestion.embedder import embed_query
from ingestion.vector_store import query_collection
from config import TOP_K_DENSE

def dense_retrieve(
    query: str,
    n: int = TOP_K_DENSE,
    where_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform dense cosine similarity retrieval using bge-small embeddings on ChromaDB.
    Returns: list of {"text": ..., "metadata": ..., "score": ..., "rank": ...}
    """
    query_emb = embed_query(query)
    docs, metas, distances = query_collection(
        query_embedding=query_emb,
        n_results=n,
        where_filter=where_filter
    )

    results = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # In cosine distance: score = 1 - distance
        score = 1.0 - float(dist)
        results.append({
            "text": doc,
            "metadata": meta,
            "score": score,
            "rank": rank,
            "chunk_id": meta.get("chunk_id", f"dense_chunk_{rank}")
        })

    return results
