"""Verify generated research reports stay separate from maintained guides."""
import json

from eval import run_experiment


def test_report_stage_creates_generated_document_directory(tmp_path, monkeypatch):
    judged_path = tmp_path / "judged_outputs.jsonl"
    report_path = tmp_path / "docs" / "generated" / "results_table.md"
    judged_path.write_text(
        json.dumps({
            "category": "answerable",
            "variants": {
                variant: {"mode": "answer", "was_hallucination": False}
                for variant in ("always_answer", "fixed_threshold", "calibrated")
            },
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_experiment, "JUDGED_OUTPUT_PATH", judged_path)
    monkeypatch.setattr(run_experiment, "REPORT_OUTPUT_PATH", report_path)

    run_experiment.report_stage()

    report = report_path.read_text(encoding="utf-8")
    assert "Coverage (95% CI)" in report
    for variant in ("always_answer", "fixed_threshold", "calibrated"):
        assert f"| answerable | {variant} |" in report
