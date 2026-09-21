"""CI regression gate: scores a small fixed subset of the golden dataset
(tests/fixtures/ci_regression_set.json - 5 examples known to score cleanly in
the v2 Contextual Retrieval baseline) against the current pipeline, and fails
if scores have dropped meaningfully from the checked-in baseline
(tests/fixtures/ci_baseline_scores.json).

Deliberately NOT a re-run of the full 437-example eval: the free-tier Gemini
quota that gated this whole project can't support a full eval on every push,
and DeepEval's own judge-model noise (malformed JSON, occasional low scores
on an otherwise-fine answer) means a single small run is noisy by nature.
Averaging deltas across all 5 questions x 4 metrics (20 numbers) smooths most
of that out while still catching a real regression (e.g. a chunking change
that breaks retrieval, or a broken prompt) rather than every case in
isolation.

Usage:
    python -m src.eval.check_regression --api-key-env GEMINI_API_KEY
"""
import argparse
import json
import sys
from pathlib import Path

from .run_deepeval import main as run_eval

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
DATASET_PATH = FIXTURES_DIR / "ci_regression_set.json"
BASELINE_PATH = FIXTURES_DIR / "ci_baseline_scores.json"
CI_REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_report_ci.json"

# Absolute drop in a metric's average score (0-1 scale) that counts as a
# regression, not noise. Loose enough to tolerate typical judge-model
# flakiness (a handful of examples can swing ~0.5 points on a single metric
# even with no code changes) but still catches a real break, where every
# question in a metric tends to collapse together (e.g. Contextual
# Precision/Recall dropping toward 0 if retrieval itself is broken).
REGRESSION_THRESHOLD = 0.20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    args = parser.parse_args()

    if CI_REPORT_PATH.exists():
        CI_REPORT_PATH.unlink()

    run_eval(
        batch_size=5,
        api_key_env=args.api_key_env,
        report_path=CI_REPORT_PATH,
        dataset_path=DATASET_PATH,
        check_shared_report=False,
    )

    current = json.loads(CI_REPORT_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current_by_q = {x["input"]: x for x in current}

    metric_deltas: dict[str, list[float]] = {}
    failed_hard = []
    for question, baseline_scores in baseline.items():
        entry = current_by_q.get(question)
        if entry is None or not entry.get("metrics"):
            failed_hard.append(f"'{question[:50]}' didn't score at all this run: {entry.get('error') if entry else 'missing'}")
            continue
        current_scores = {m["name"]: m["score"] for m in entry["metrics"]}
        for name, base_score in baseline_scores.items():
            delta = current_scores.get(name, 0.0) - base_score
            metric_deltas.setdefault(name, []).append(delta)

    print(f"{'Metric':<25} {'Baseline avg':>12} {'Current avg':>12} {'Delta':>8}")
    regressed = []
    for name, deltas in metric_deltas.items():
        base_avg = sum(baseline[q][name] for q in baseline if name in baseline[q]) / len(deltas)
        cur_avg = base_avg + sum(deltas) / len(deltas)
        avg_delta = sum(deltas) / len(deltas)
        flag = " <-- REGRESSION" if avg_delta < -REGRESSION_THRESHOLD else ""
        print(f"{name:<25} {base_avg:>12.3f} {cur_avg:>12.3f} {avg_delta:>+8.3f}{flag}")
        if avg_delta < -REGRESSION_THRESHOLD:
            regressed.append(name)

    if failed_hard:
        print("\nQuestions that failed to score at all:")
        for msg in failed_hard:
            print(f"  - {msg}")

    if regressed or failed_hard:
        print(f"\nFAIL: regression detected in {regressed or 'scoring itself'}.")
        return 1

    print("\nPASS: no metric dropped more than the noise tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
