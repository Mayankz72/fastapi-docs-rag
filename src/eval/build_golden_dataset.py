"""Turn fetched GitHub discussions into a golden eval set.

Filters out pairs that are too short/noisy to be a fair RAG question, and caps
answer length so the "expected output" stays a reasonable comparison target.
"""
import json
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "discussions.json"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "golden_dataset.json"

MIN_QUESTION_CHARS = 20
MIN_ANSWER_CHARS = 40
MAX_ANSWER_CHARS = 2000


def build() -> list[dict]:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    golden = []
    for pair in raw:
        q = pair["question"].strip()
        a = pair["answer"].strip()
        if len(q) < MIN_QUESTION_CHARS or len(a) < MIN_ANSWER_CHARS:
            continue
        golden.append(
            {
                "question": q,
                "expected_answer": a[:MAX_ANSWER_CHARS],
                "url": pair["url"],
            }
        )
    return golden


if __name__ == "__main__":
    dataset = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Built golden dataset: {len(dataset)} examples -> {OUT_PATH}")
