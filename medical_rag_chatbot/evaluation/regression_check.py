import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.eval_retrieval import load_golden_dataset, run_retrieval_config

BASELINE_FILE = BASE_DIR / "evaluation" / "baseline_scores.json"

def run_regression_test(threshold_percent: float = 5.0) -> bool:
    """
    Compare current retrieval scores against baseline_scores.json.
    Flags a regression if any metric drops by more than threshold_percent (default: 5.0%).
    """
    print("=== Running Regression Test Against Baseline ===")
    if not BASELINE_FILE.exists():
        print(f"No baseline file found at {BASELINE_FILE}. Creating one now...")
        from evaluation.eval_retrieval import evaluate_all
        evaluate_all()
        return True

    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    questions = load_golden_dataset()
    current = run_retrieval_config("full_pipeline", questions, top_k=3)

    print(f"\nBaseline Metrics ({baseline.get('Configuration', 'baseline')}):")
    for k in ["Hit Rate@3", "MRR", "Precision@3"]:
        print(f"  {k}: {baseline.get(k, 0):.4f}")

    print(f"\nCurrent Pipeline Metrics:")
    for k in ["Hit Rate@3", "MRR", "Precision@3"]:
        print(f"  {k}: {current.get(k, 0):.4f}")

    regressions = []
    for metric in ["Hit Rate@3", "MRR", "Precision@3"]:
        b_val = baseline.get(metric, 0.0)
        c_val = current.get(metric, 0.0)
        if b_val > 0:
            pct_change = ((c_val - b_val) / b_val) * 100.0
            print(f"  {metric} delta: {pct_change:+.2f}%")
            if pct_change < -threshold_percent:
                regressions.append((metric, b_val, c_val, pct_change))

    if regressions:
        print("\n[WARNING] REGRESSION DETECTED!")
        for metric, b_val, c_val, change in regressions:
            print(f"  ❌ {metric} dropped from {b_val:.4f} to {c_val:.4f} ({change:.2f}%)")
        return False
    else:
        print("\n[PASS] No regressions detected! All retrieval benchmarks within safe margins.")
        return True

if __name__ == "__main__":
    success = run_regression_test()
    sys.exit(0 if success else 1)
