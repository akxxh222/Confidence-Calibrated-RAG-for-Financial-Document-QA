"""Merge checkpointed pipeline runs with manually reviewed binary labels."""

import json
from pathlib import Path


RUNS_PATH = Path("eval/calibration_seed_runs.jsonl")
REVIEWS_PATH = Path("eval/calibration_seed_reviews.jsonl")
OUTPUT_PATH = Path("eval/labeled_seed.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_labeled_rows(runs: list[dict], reviews: list[dict]) -> list[dict]:
    run_by_id = {row["question_id"]: row for row in runs}
    if len(run_by_id) != len(runs):
        raise ValueError("Duplicate question IDs in calibration runs")

    review_by_id = {row["question_id"]: row for row in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("Duplicate question IDs in calibration reviews")

    unknown = sorted(review_by_id.keys() - run_by_id.keys())
    if unknown:
        raise ValueError(f"Reviews without completed runs: {', '.join(unknown)}")

    labeled = []
    for run in runs:
        review = review_by_id.get(run["question_id"])
        if review is None:
            continue
        if review.get("label") not in (0, 1):
            raise ValueError(f"{run['question_id']}: label must be 0 or 1")
        for field in ("label_rationale", "reviewer", "reviewed_on"):
            if not review.get(field):
                raise ValueError(f"{run['question_id']}: missing {field}")

        labeled.append({
            "question_id": run["question_id"],
            "category": run["category"],
            "question_text": run["question_text"],
            "expected_answer": run["expected_answer"],
            "generated_answer": run["generated_answer"],
            "top1_score": run["top1_score"],
            "spread": run["spread"],
            "self_consistency_score": run["self_consistency_score"],
            "label": review["label"],
            "label_rationale": review["label_rationale"],
            "reviewer": review["reviewer"],
            "reviewed_on": review["reviewed_on"],
            "generation_model": run["generation_model"],
            "embedding_model": run["embedding_model"],
            "sample_count": run["sample_count"],
            "retrieved_chunks": run["retrieved_chunks"],
        })
    return labeled


def main() -> None:
    rows = build_labeled_rows(load_jsonl(RUNS_PATH), load_jsonl(REVIEWS_PATH))
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    positives = sum(row["label"] for row in rows)
    print(f"Wrote {len(rows)} reviewed rows: {positives} correct, "
          f"{len(rows) - positives} incorrect")


if __name__ == "__main__":
    main()
