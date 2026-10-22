import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from app.calibration.train import load_labeled_data, validate_training_data
from eval.build_labeled_seed import build_labeled_rows
from eval.dataset import load_and_validate


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_row(label=1):
    return {
        "top1_score": 0.8,
        "spread": 0.05,
        "self_consistency_score": 1.0,
        "label": label,
    }


def test_load_labeled_data_returns_fixed_feature_order(tmp_path):
    path = tmp_path / "seed.jsonl"
    write_rows(path, [valid_row()])

    features, labels = load_labeled_data(str(path))

    assert features.tolist() == [[0.8, 0.05, 1.0]]
    assert labels.tolist() == [1]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top1_score", float("nan"), "finite"),
        ("spread", -0.1, "spread"),
        ("self_consistency_score", 1.1, "self_consistency_score"),
        ("label", 2, "label"),
    ],
)
def test_load_labeled_data_rejects_invalid_rows(
    tmp_path, field, value, message
):
    path = tmp_path / "seed.jsonl"
    row = valid_row()
    row[field] = value
    write_rows(path, [row])

    with pytest.raises(ValueError, match=message):
        load_labeled_data(str(path))


def test_training_data_requires_enough_rows_from_both_classes():
    with pytest.raises(ValueError, match="at least 30 rows"):
        validate_training_data(np.asarray([0] * 20 + [1] * 9))


def test_training_data_accepts_balanced_seed():
    validate_training_data(np.asarray([0] * 15 + [1] * 15))


def test_calibration_questions_are_unique_balanced_and_test_disjoint():
    path = Path("eval/calibration_seed_questions.jsonl")
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    final_questions = load_and_validate(Path("eval/test_set"))
    texts = [row["question_text"].strip().casefold() for row in questions]
    final_texts = {
        row["question_text"].strip().casefold() for row in final_questions
    }
    counts = Counter(row["category"] for row in questions)

    assert len(questions) >= 30
    assert len(texts) == len(set(texts))
    assert set(texts).isdisjoint(final_texts)
    assert counts["answerable"] >= 10
    assert counts["false_premise"] >= 5
    assert counts["unanswerable"] >= 10


def test_build_labeled_rows_merges_review_provenance():
    run = {
        "question_id": "seed_ans_001",
        "category": "answerable",
        "question_text": "What was revenue?",
        "expected_answer": "$10 million",
        "generated_answer": "Revenue was $10 million.",
        "top1_score": 0.8,
        "spread": 0.1,
        "self_consistency_score": 1.0,
        "generation_model": "gemini-test",
        "embedding_model": "embedding-test",
        "sample_count": 5,
        "retrieved_chunks": [],
    }
    review = {
        "question_id": "seed_ans_001",
        "label": 1,
        "label_rationale": "Exact supported answer.",
        "reviewer": "Reviewer",
        "reviewed_on": "2026-10-19",
    }

    rows = build_labeled_rows([run], [review])

    assert rows[0]["label"] == 1
    assert rows[0]["reviewer"] == "Reviewer"


def test_build_labeled_rows_rejects_review_without_run():
    review = {
        "question_id": "missing",
        "label": 0,
        "label_rationale": "No run.",
        "reviewer": "Reviewer",
        "reviewed_on": "2026-10-19",
    }

    with pytest.raises(ValueError, match="without completed runs"):
        build_labeled_rows([], [review])
