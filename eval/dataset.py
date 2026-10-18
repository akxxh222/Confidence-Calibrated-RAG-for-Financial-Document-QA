"""Load and validate the hand-reviewed financial QA evaluation set."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CATEGORIES = {"answerable", "false_premise", "unanswerable"}
PLACEHOLDER_RE = re.compile(r"placeholder|replace_with|\btbd\b|\btodo\b", re.I)
REQUIRED_FIELDS = {
    "question_id",
    "question_text",
    "category",
    "company",
    "doc_type",
    "fiscal_period",
    "ground_truth_answer",
    "evidence",
    "absence_check",
    "review",
}
REVIEW_FIELDS = {
    "author_verified",
    "author_reviewed_on",
    "independent_verified",
    "independent_reviewer",
    "independent_reviewed_on",
    "disagreement",
    "resolution",
}


def _has_placeholder(value) -> bool:
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_placeholder(item) for item in value)
    return False


def validate_records(records: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = Counter(row.get("question_id") for row in records)
    texts = Counter(
        str(row.get("question_text", "")).strip().casefold() for row in records
    )

    for index, row in enumerate(records, start=1):
        label = row.get("question_id") or f"row {index}"
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue

        if ids[row["question_id"]] > 1:
            errors.append(f"{label}: duplicate question_id")
        normalized_text = str(row["question_text"]).strip().casefold()
        if texts[normalized_text] > 1:
            errors.append(f"{label}: duplicate question_text")
        if row["category"] not in CATEGORIES:
            errors.append(f"{label}: invalid category {row['category']!r}")
        if _has_placeholder(row):
            errors.append(f"{label}: placeholder text is not allowed")

        review = row.get("review")
        if (
            not isinstance(review, dict)
            or review.get("author_verified") is not True
        ):
            errors.append(f"{label}: author review is required")
        else:
            missing_review = sorted(REVIEW_FIELDS - review.keys())
            if missing_review:
                errors.append(
                    f"{label}: missing review fields: "
                    f"{', '.join(missing_review)}"
                )
            if not isinstance(review.get("independent_verified"), bool):
                errors.append(
                    f"{label}: independent_verified must be boolean"
                )
            if review.get("independent_verified") and not all(
                review.get(name)
                for name in (
                    "independent_reviewer",
                    "independent_reviewed_on",
                )
            ):
                errors.append(
                    f"{label}: completed independent review requires "
                    "reviewer and date"
                )
            if review.get("disagreement") and not review.get("resolution"):
                errors.append(
                    f"{label}: disagreement requires a resolution"
                )

        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{label}: evidence must be a list")
            evidence = []

        if row["category"] in {"answerable", "false_premise"}:
            if not row.get("ground_truth_answer"):
                errors.append(
                    f"{label}: category requires a ground_truth_answer"
                )
            if not evidence:
                errors.append(f"{label}: category requires evidence")
            for item in evidence:
                evidence_fields = ("source_url", "section", "excerpt")
                if not isinstance(item, dict) or not all(
                    item.get(name) for name in evidence_fields
                ):
                    errors.append(
                        f"{label}: evidence requires source_url, section, "
                        "and excerpt"
                    )
        else:
            if row.get("ground_truth_answer") is not None:
                errors.append(
                    f"{label}: unanswerable ground_truth_answer must be null"
                )
            absence = row.get("absence_check")
            absence_fields = {
                "scope",
                "method",
                "search_terms",
                "result",
                "checked_on",
            }
            if not isinstance(absence, dict) or not all(
                absence.get(name) for name in absence_fields
            ):
                errors.append(
                    f"{label}: unanswerable category requires absence_check"
                )

    return errors


def load_jsonl(directory: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                message = f"{path.name}:{line_number}: {exc.msg}"
                raise ValueError(message) from exc
    return records


def load_and_validate(directory: Path) -> list[dict]:
    records = load_jsonl(directory)
    errors = validate_records(records)
    if errors:
        message = "Invalid evaluation dataset:\n- " + "\n- ".join(errors)
        raise ValueError(message)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="eval/test_set")
    args = parser.parse_args()

    records = load_and_validate(Path(args.directory))
    counts = Counter(row["category"] for row in records)
    independently_verified = sum(
        row["review"]["independent_verified"] for row in records
    )
    print(f"Valid records: {len(records)}")
    for category in sorted(CATEGORIES):
        print(f"  {category}: {counts[category]}")
    print(f"Independently verified: {independently_verified}/{len(records)}")


if __name__ == "__main__":
    main()
