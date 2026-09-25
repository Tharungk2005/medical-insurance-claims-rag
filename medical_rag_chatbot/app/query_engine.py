import time
from typing import List, Dict, Any, Optional

from retrieval.cache import get_cached, set_cache
from retrieval.reranker import retrieve_and_rerank
from llm.prompts import build_prompt
from llm.llm_caller import call_llm, call_llava

def get_confidence_tier(score: float) -> str:
    """Map normalized reranker confidence score to user-friendly tier."""
    if score >= 0.70:
        return "High"
    elif score >= 0.40:
        return "Medium"
    return "Low"

def answer_query(
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    use_cot: bool = False,
    variant: str = "production"
) -> Dict[str, Any]:
    """
    Main query pipeline orchestrating caching, hybrid retrieval, cross-encoder reranking,
    prompt assembly, and LLM generation.
    Returns:
    {
        "answer": str,
        "sources": List[dict],
        "cache_hit": bool,
        "cache_type": str,
        "confidence": str,
        "max_score": float,
        "latency_sec": float
    }
    """
    start_time = time.time()
    if chat_history is None:
        chat_history = []

    # 1. Check exact and semantic cache (only when no restrictive filter is active)
    if not metadata_filter:
        cached_result = get_cached(query)
        if cached_result:
            latency = time.time() - start_time
            return {
                "answer": cached_result["answer"],
                "sources": cached_result.get("sources", []),
                "cache_hit": True,
                "cache_type": cached_result.get("cache_type", "exact"),
                "confidence": "High",
                "max_score": 1.0,
                "latency_sec": latency
            }

    # 2. Hybrid Retrieval + Cross-Encoder Reranking
    retrieval_res = retrieve_and_rerank(query, where_filter=metadata_filter)
    chunks = retrieval_res["chunks"]
    insufficient = retrieval_res["insufficient"]
    max_score = retrieval_res["max_score"]

    if insufficient or not chunks:
        latency = time.time() - start_time
        return {
            "answer": "I do not have enough information in the available documents to answer this.",
            "sources": [],
            "cache_hit": False,
            "cache_type": "none",
            "confidence": "Low",
            "max_score": max_score,
            "latency_sec": latency
        }

    # 3. Vision check: If primary evidence is an image with an image_path, optionally query LLaVA
    first_chunk = chunks[0]
    if first_chunk.get("metadata", {}).get("modality") == "image":
        img_path = first_chunk.get("metadata", {}).get("image_path")
        if img_path:
            llava_text = call_llava(img_path, query)
            if llava_text:
                # Augment first chunk with direct vision output
                first_chunk["text"] += f"\nVision Analysis: {llava_text}"

    # 4. Assemble Prompt with Token Budget Management
    messages = build_prompt(
        query=query,
        retrieved_chunks=chunks,
        chat_history=chat_history,
        variant=variant,
        use_cot=use_cot
    )

    # 5. Call LLM
    answer = call_llm(messages)

    # 6. Store in Cache (for non-filtered queries)
    if not metadata_filter:
        try:
            set_cache(query=query, answer=answer, sources=chunks)
        except Exception:
            pass

    latency = time.time() - start_time
    confidence_tier = get_confidence_tier(max_score)

    return {
        "answer": answer,
        "sources": chunks,
        "cache_hit": False,
        "cache_type": "none",
        "confidence": confidence_tier,
        "max_score": max_score,
        "latency_sec": latency
    }
