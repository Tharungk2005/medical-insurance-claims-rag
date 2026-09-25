import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import BM25_INDEX_PATH

_cached_bm25_data = None

def build_bm25_index(chunks: List[Document], save_path: str | Path = BM25_INDEX_PATH) -> bool:
    """
    Build BM25 index over all document chunks and serialize with pickle.
    """
    global _cached_bm25_data
    if not chunks:
        return False

    os.makedirs(Path(save_path).parent, exist_ok=True)
    
    # Tokenize chunk texts
    corpus = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)

    data = {
        "bm25": bm25,
        "chunks": chunks
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    _cached_bm25_data = data
    return True

def load_bm25_index(index_path: str | Path = BM25_INDEX_PATH) -> Optional[Dict[str, Any]]:
    """Load BM25 index from disk."""
    global _cached_bm25_data
    if _cached_bm25_data is not None:
        return _cached_bm25_data

    path = Path(index_path)
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            _cached_bm25_data = data
            return data
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        return None

def bm25_retrieve(query: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve top-N chunks using BM25 keyword scoring.
    Returns: list of {"text": ..., "metadata": ..., "score": ..., "rank": ...}
    """
    data = load_bm25_index()
    if data is None or not data.get("chunks"):
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Rank indices by score descending
    num_results = min(n, len(chunks))
    sorted_indices = scores.argsort()[::-1][:num_results]

    results = []
    for rank, idx in enumerate(sorted_indices, start=1):
        score_val = float(scores[idx])
        results.append({
            "text": chunks[idx].page_content,
            "metadata": chunks[idx].metadata,
            "score": score_val,
            "rank": rank
        })

    return results
