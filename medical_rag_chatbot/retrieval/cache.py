import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config import CACHE_FILE, CACHE_TTL_SECONDS, CACHE_SIMILARITY_THRESHOLD

def _ensure_cache_dir():
    os.makedirs(Path(CACHE_FILE).parent, exist_ok=True)

def load_cache() -> Dict[str, Any]:
    """Load query cache from disk."""
    _ensure_cache_dir()
    if not Path(CACHE_FILE).exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: Dict[str, Any]) -> None:
    """Save query cache to disk."""
    _ensure_cache_dir()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def hash_query(query: str) -> str:
    """Compute MD5 hash of standardized query string."""
    return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()

def get_exact_cache(query: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached answer if exact query hash matches and TTL has not expired.
    """
    cache = load_cache()
    key = hash_query(query)
    entry = cache.get(key)
    if not entry:
        return None

    timestamp = entry.get("timestamp", 0)
    if time.time() - timestamp > CACHE_TTL_SECONDS:
        # Expired
        return None

    return {
        "answer": entry.get("answer", ""),
        "sources": entry.get("sources", []),
        "cache_type": "exact",
        "cache_hit": True
    }

def get_semantic_cache(query: str, query_embedding: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached answer if semantically equivalent past query has cosine similarity > 0.95.
    """
    cache = load_cache()
    if not cache:
        return None

    if query_embedding is None:
        from ingestion.embedder import embed_query
        query_embedding = embed_query(query)

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return None

    best_entry = None
    best_sim = -1.0

    current_time = time.time()
    for key, item in cache.items():
        if current_time - item.get("timestamp", 0) > CACHE_TTL_SECONDS:
            continue
        cached_emb = item.get("embedding")
        if not cached_emb:
            continue

        c_vec = np.array(cached_emb, dtype=np.float32)
        c_norm = np.linalg.norm(c_vec)
        if c_norm == 0:
            continue

        sim = float(np.dot(query_embedding, c_vec) / (query_norm * c_norm))
        if sim > best_sim:
            best_sim = sim
            best_entry = item

    if best_sim >= CACHE_SIMILARITY_THRESHOLD and best_entry is not None:
        return {
            "answer": best_entry.get("answer", ""),
            "sources": best_entry.get("sources", []),
            "cache_type": "semantic",
            "similarity": best_sim,
            "cache_hit": True
        }

    return None

def get_cached(query: str) -> Optional[Dict[str, Any]]:
    """Check exact cache first, then semantic cache."""
    exact = get_exact_cache(query)
    if exact:
        return exact
    return get_semantic_cache(query)

def set_cache(query: str, answer: str, sources: Optional[list] = None, embedding: Optional[np.ndarray] = None) -> None:
    """Store query, embedding, answer, sources, and timestamp into cache."""
    cache = load_cache()
    key = hash_query(query)

    if embedding is None:
        try:
            from ingestion.embedder import embed_query
            embedding = embed_query(query)
        except Exception:
            embedding = None

    emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    cache[key] = {
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "embedding": emb_list,
        "timestamp": time.time()
    }
    save_cache(cache)

def invalidate_cache() -> None:
    """Flush and invalidate all cache entries."""
    if Path(CACHE_FILE).exists():
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass
    save_cache({})
