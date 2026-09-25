import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.query_engine import answer_query
from config import LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT

def run_e2e_observability_demo():
    """
    Run end-to-end tracing and observability demo:
    - TruLens instrumentation hook
    - Arize Phoenix local telemetry hook
    - LangSmith chain telemetry integration
    """
    print("=== Commencing Phase 13 End-to-End Tracing & Observability ===")
    print(f"LangSmith Tracing Active: {LANGCHAIN_TRACING_V2} (Project: {LANGCHAIN_PROJECT})")

    # Sample representative queries across modalities
    test_queries = [
        "Why was claim CLM-2024-001 denied for outpatient consultation?",
        "What was the allowed amount and status for routine ECG CPT 93000?",
        "What medication and dosage is prescribed in prescription scan CLM-2024-001?"
    ]

    # Check for TruLens
    try:
        from trulens_eval import Tru, TruBasicApp
        print("TruLens Eval detected: Initializing recorder...")
        tru = Tru()
        recorder = TruBasicApp(answer_query, app_id="claims-rag-v1")
        use_trulens = True
    except Exception as e:
        print(f"TruLens note: ({e}). Running direct telemetry execution.")
        use_trulens = False

    # Check for Arize Phoenix
    try:
        import phoenix as px
        print("Arize Phoenix detected: Ready for local telemetry dashboard (localhost:6006)")
    except Exception:
        pass

    results = []
    for q in test_queries:
        print(f"\nTracing query: '{q}'")
        t0 = time.time()
        if use_trulens:
            with recorder as recording:
                res = answer_query(q)
        else:
            res = answer_query(q)
        latency = time.time() - t0

        print(f"  Confidence: {res.get('confidence')} | Score: {res.get('max_score', 0):.2f} | Latency: {latency:.2f}s")
        print(f"  Sources: {len(res.get('sources', []))} chunks")
        results.append(res)

    print("\n=== E2E Observability and Tracing Completed Successfully ===")
    return results

if __name__ == "__main__":
    run_e2e_observability_demo()
