"""Recompute calibration consistency features from saved generation samples."""

import json
from pathlib import Path

from app.generation.self_consistency import score_samples


RUNS_PATH = Path("eval/calibration_seed_runs.jsonl")


def main() -> None:
    rows = [
        json.loads(line)
        for line in RUNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        result = score_samples(row["samples"])
        if (
            row["self_consistency_score"] != result.self_consistency_score
            or row["extracted_numbers"] != result.extracted_numbers
        ):
            changed += 1
        row["self_consistency_score"] = result.self_consistency_score
        row["extracted_numbers"] = result.extracted_numbers

    RUNS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Rescored {len(rows)} rows; {changed} feature rows changed")


if __name__ == "__main__":
    main()
