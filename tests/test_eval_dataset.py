import json
from collections import Counter
from pathlib import Path

import pytest

from eval.dataset import load_and_validate, validate_records


def evidence():
    return {
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1/"
            "example-index.html"
        ),
        "section": "financial_statements",
        "excerpt": "Total net sales were $416,161 million.",
    }


def record(question_id="ans_001", question_text="What were total net sales?"):
    return {
        "question_id": question_id,
        "question_text": question_text,
        "category": "answerable",
        "company": "AAPL",
        "doc_type": "10-K",
        "fiscal_period": "2025-09-27",
        "ground_truth_answer": "$416,161 million",
        "evidence": [evidence()],
        "absence_check": None,
        "review": {
            "author_verified": True,
            "author_reviewed_on": "2026-10-18",
            "independent_verified": False,
            "independent_reviewer": None,
            "independent_reviewed_on": None,
            "disagreement": None,
            "resolution": None,
        },
    }


def test_valid_answerable_record_passes():
    assert validate_records([record()]) == []


def test_duplicate_question_text_is_rejected():
    rows = [record(), record("ans_002")]

    errors = validate_records(rows)

    assert any("duplicate question_text" in error for error in errors)


def test_placeholders_and_missing_evidence_are_rejected():
    row = record()
    row["ground_truth_answer"] = "REPLACE_WITH_VERIFIED_FIGURE"
    row["evidence"] = []

    errors = validate_records([row])

    assert any("placeholder" in error for error in errors)
    assert any("requires evidence" in error for error in errors)


def test_unanswerable_requires_corpus_absence_check():
    row = record("unans_001", "What was the private forecast?")
    row.update({
        "category": "unanswerable",
        "ground_truth_answer": None,
        "evidence": [],
        "absence_check": None,
    })

    errors = validate_records([row])

    assert any("absence_check" in error for error in errors)


def test_completed_independent_review_requires_reviewer_and_date():
    row = record()
    row["review"]["independent_verified"] = True

    errors = validate_records([row])

    assert any("reviewer and date" in error for error in errors)


def test_disagreement_requires_resolution():
    row = record()
    row["review"]["disagreement"] = "The period appears incorrect."

    errors = validate_records([row])

    assert any("requires a resolution" in error for error in errors)


def test_load_and_validate_reads_all_jsonl_files(tmp_path):
    (tmp_path / "answerable.jsonl").write_text(
        json.dumps(record()) + "\n", encoding="utf-8"
    )

    rows = load_and_validate(tmp_path)

    assert [row["question_id"] for row in rows] == ["ans_001"]


def test_load_and_validate_raises_with_line_context(tmp_path):
    path = tmp_path / "answerable.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="answerable.jsonl:1"):
        load_and_validate(tmp_path)


def test_project_evaluation_set_meets_milestone_three_size_and_balance():
    rows = load_and_validate(Path("eval/test_set"))
    counts = Counter(row["category"] for row in rows)

    assert 70 <= len(rows) <= 100
    assert 30 <= counts["answerable"] <= 40
    assert 20 <= counts["false_premise"] <= 30
    assert 20 <= counts["unanswerable"] <= 30


def test_project_evidence_uses_stable_sec_filing_urls():
    rows = load_and_validate(Path("eval/test_set"))
    evidence = [item for row in rows for item in row["evidence"]]

    assert evidence
    assert all(
        item["source_url"].startswith("https://www.sec.gov/Archives/")
        and item["source_url"].endswith("-index.html")
        for item in evidence
    )
