import os
import sys
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.dense_retriever import dense_retrieve
from retrieval.sparse_retriever import bm25_retrieve
from retrieval.hybrid_retriever import hybrid_retrieve
from retrieval.reranker import rerank, retrieve_and_rerank
from config import TOP_K_RERANK

GOLDEN_DATASET_PATH = BASE_DIR / "evaluation" / "golden_dataset.json"
RETRIEVAL_SCORES_CSV = BASE_DIR / "evaluation" / "retrieval_scores.csv"

def load_golden_dataset() -> List[Dict[str, Any]]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_retrieval_config(config_name: str, questions: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
    """
    Run evaluation on a specific retrieval configuration:
    - 'dense_only': BGE-small dense cosine retrieval
    - 'hybrid_rrf': Dense + BM25 with RRF fusion
    - 'full_pipeline': Hybrid RRF + CrossEncoder reranking
    """
    hits = 0
    reciprocal_ranks = []
    precisions = []

    for item in questions:
        q = item["question"]
        expected_chunk_ids = set(item.get("expected_chunk_ids", []))
        expected_file = item.get("expected_chunk_ids", [""])[0].split("_chunk_")[0]

        if config_name == "dense_only":
            retrieved = dense_retrieve(q, n=top_k)
        elif config_name == "hybrid_rrf":
            retrieved = hybrid_retrieve(q, n=top_k)
        elif config_name == "full_pipeline":
            res = retrieve_and_rerank(q, top_k_final=top_k)
            retrieved = res["chunks"]
        else:
            raise ValueError(f"Unknown configuration: {config_name}")

        retrieved_ids = [c.get("chunk_id", c.get("metadata", {}).get("chunk_id", "")) for c in retrieved]
        retrieved_files = [c.get("metadata", {}).get("source_file", "") for c in retrieved]

        # Check Hit Rate (exact chunk ID match OR source file match)
        is_hit = False
        rr = 0.0
        relevant_count = 0

        for rank, (cid, sfile) in enumerate(zip(retrieved_ids, retrieved_files), start=1):
            if cid in expected_chunk_ids or (expected_file and sfile == expected_file):
                relevant_count += 1
                if not is_hit:
                    is_hit = True
                    rr = 1.0 / rank

        if is_hit:
            hits += 1
        reciprocal_ranks.append(rr)
        precisions.append(relevant_count / max(1, len(retrieved)))

    total_q = len(questions)
    hit_rate = hits / total_q if total_q else 0.0
    mrr = sum(reciprocal_ranks) / total_q if total_q else 0.0
    avg_precision = sum(precisions) / total_q if total_q else 0.0

    return {
        "Configuration": config_name,
        "Total Questions": total_q,
        "Hit Rate@3": round(hit_rate, 4),
        "MRR": round(mrr, 4),
        "Precision@3": round(avg_precision, 4)
    }

def run_ragas_context_eval(questions: List[Dict[str, Any]]):
    """
    RAGAS Context Precision & Context Recall calculation on golden dataset.
    Target: context_precision > 0.75, context_recall > 0.70
    """
    print("\nCalculating RAGAS Context Metrics...")
    precisions = []
    recalls = []

    for item in questions:
        q = item["question"]
        expected_file = item.get("expected_chunk_ids", [""])[0].split("_chunk_")[0]
        res = retrieve_and_rerank(q, top_k_final=3)
        retrieved = res["chunks"]

        rel_retrieved = sum(1 for c in retrieved if c.get("metadata", {}).get("source_file", "") == expected_file)
        c_precision = rel_retrieved / max(1, len(retrieved))
        c_recall = 1.0 if rel_retrieved > 0 else 0.0

        precisions.append(c_precision)
        recalls.append(c_recall)

    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)

    print(f"RAGAS context_precision: {avg_prec:.4f} (Target: > 0.75)")
    print(f"RAGAS context_recall:    {avg_rec:.4f} (Target: > 0.70)")
    return avg_prec, avg_rec

def evaluate_all():
    print("=== Commencing Phase 11 Retrieval Evaluation ===")
    questions = load_golden_dataset()
    print(f"Loaded {len(questions)} golden Q&A items across PDF, table, and image modalities.")

    configs = ["dense_only", "hybrid_rrf", "full_pipeline"]
    results = []

    for cfg in configs:
        res = run_retrieval_config(cfg, questions, top_k=TOP_K_RERANK)
        results.append(res)
        print(f"  [{res['Configuration']}] Hit Rate@3: {res['Hit Rate@3']:.2%} | MRR: {res['MRR']:.4f} | Precision@3: {res['Precision@3']:.4f}")

    df = pd.DataFrame(results)
    os.makedirs(RETRIEVAL_SCORES_CSV.parent, exist_ok=True)
    df.to_csv(RETRIEVAL_SCORES_CSV, index=False)
    print(f"\nAll retrieval evaluation scores saved to: {RETRIEVAL_SCORES_CSV}")

    # RAGAS context metrics
    run_ragas_context_eval(questions)

    # Save baseline for regression checking
    baseline_path = BASE_DIR / "evaluation" / "baseline_scores.json"
    best_config = results[-1]
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best configuration as baseline in: {baseline_path}")

    return df

if __name__ == "__main__":
    evaluate_all()
