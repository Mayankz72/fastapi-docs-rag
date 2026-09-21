"""Merge one or more shard report files (from run_deepeval.py --report-path)
into the main eval_report.json, deduping by question. Safe to re-run - existing
entries in the main report always win over a shard's copy of the same question."""
import json
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_report.json"


def main(shard_paths: list[str]) -> None:
    main_report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else []
    seen = {entry["input"] for entry in main_report}

    added = 0
    for shard_path in shard_paths:
        path = Path(shard_path)
        if not path.exists():
            print(f"skip {path} (does not exist yet)")
            continue
        shard_report = json.loads(path.read_text(encoding="utf-8"))
        for entry in shard_report:
            if entry["input"] not in seen:
                main_report.append(entry)
                seen.add(entry["input"])
                added += 1

    REPORT_PATH.write_text(json.dumps(main_report, indent=2), encoding="utf-8")
    print(f"Merged {added} new entries -> {REPORT_PATH} ({len(main_report)} total scored)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.eval.merge_shard_reports <shard_report.json> [more shard files...]")
        sys.exit(1)
    main(sys.argv[1:])
