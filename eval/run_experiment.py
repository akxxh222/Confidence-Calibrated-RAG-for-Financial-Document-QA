"""
Runs all three system variants (always_answer, fixed_threshold, calibrated)
against the hand-verified 3-category test set and produces the Section 12
comparison table.

This script does NOT auto-judge hallucination — that requires a human (or a
carefully-designed, separately-validated LLM-judge you report on explicitly)
comparing the generated answer to ground_truth_answer. This script's job is
to run the systems, log confidence signals, and save results to eval_results
for that judgment pass, then reload judged results to compute the table.

Usage:
    python -m eval.run_experiment --stage generate   # run systems, save raw outputs
    python -m eval.run_experiment --stage report      # after hand-judging, build table
"""
import argparse
import json
from pathlib import Path

from eval.baselines import run_always_answer, run_fixed_threshold
from app.retrieval.retriever import retrieve
from app.generation.self_consistency import run_self_consistency
from app.calibration.model import CalibrationModel
from app.calibration.features import CalibrationFeatures
from eval.metrics import format_metrics_table

TEST_SET_DIR = Path("eval/test_set")
RAW_OUTPUT_PATH = Path("eval/raw_run_outputs.jsonl")
JUDGED_OUTPUT_PATH = Path("eval/judged_outputs.jsonl")  # produced by human review


def load_test_set() -> list[dict]:
    questions = []
    for path in TEST_SET_DIR.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                questions.append(json.loads(line))
    return questions


def run_calibrated(question: str, model: CalibrationModel) -> dict:
    retrieval = retrieve(question)
    if not retrieval.chunks:
        return {"mode": "refuse", "answer": "No relevant context found.",
                "retrieval_confidence": 0.0, "self_consistency_score": 0.0}
    sc = run_self_consistency(question, retrieval.chunks)
    features = CalibrationFeatures(
        top1_score=retrieval.top1_score, spread=retrieval.spread,
        self_consistency_score=sc.self_consistency_score,
    )
    prob = float(model.predict_proba([features.as_vector()])[0])
    mode = model.decide(prob)
    return {
        "mode": mode, "answer": sc.samples[0], "confidence": prob,
        "retrieval_confidence": retrieval.top1_score,
        "self_consistency_score": sc.self_consistency_score,
    }


def generate_stage():
    questions = load_test_set()
    calib_model = CalibrationModel.load()

    with open(RAW_OUTPUT_PATH, "w") as out:
        for q in questions:
            record = {
                "question_id": q["question_id"],
                "category": q["category"],
                "question_text": q["question_text"],
                "ground_truth_answer": q.get("ground_truth_answer"),
                "variants": {
                    "always_answer": run_always_answer(q["question_text"]),
                    "fixed_threshold": run_fixed_threshold(q["question_text"]),
                    "calibrated": run_calibrated(q["question_text"], calib_model),
                },
            }
            out.write(json.dumps(record) + "\n")
            print(f"Ran {q['question_id']} ({q['category']})")

    print(f"\nSaved raw outputs to {RAW_OUTPUT_PATH}.")
    print("Next: hand-judge each variant's answer against ground_truth_answer, "
          "add a 'was_hallucination' bool per variant, save to "
          f"{JUDGED_OUTPUT_PATH}, then run --stage report.")


def report_stage():
    if not JUDGED_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{JUDGED_OUTPUT_PATH} not found — hand-judge {RAW_OUTPUT_PATH} first."
        )

    by_variant_category: dict[str, dict[str, list[dict]]] = {
        "always_answer": {}, "fixed_threshold": {}, "calibrated": {}
    }
    with open(JUDGED_OUTPUT_PATH) as f:
        for line in f:
            row = json.loads(line)
            category = row["category"]
            for variant, result in row["variants"].items():
                by_variant_category[variant].setdefault(category, []).append(result)

    table = format_metrics_table(by_variant_category)
    print(table)
    Path("eval/results_table.md").write_text(table)
    print("\nSaved to eval/results_table.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "report"], required=True)
    args = parser.parse_args()
    if args.stage == "generate":
        generate_stage()
    else:
        report_stage()
