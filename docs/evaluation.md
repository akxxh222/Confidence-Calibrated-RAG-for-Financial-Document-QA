# Evaluation workflow

## Dataset separation

The reviewed development seed has 42 questions and trains the correctness
classifier. Its files are `eval/calibration_seed_questions.jsonl`,
`eval/calibration_seed_reviews.jsonl`, and `eval/labeled_seed.jsonl`.
See [Calibration](calibration.md) for labels and threshold selection.

The final evaluation set is separate: `eval/test_set/answerable.jsonl`,
`false_premise.jsonl`, and `unanswerable.jsonl`, with 30 answerable, 20
false-premise, and 25 unanswerable questions (75 total). Tests check that
normalized question texts do not overlap the
development seed. These dataset categories are not predefined response modes:
an answerable question can be hedged, and a false premise can be corrected.

## Validate and review

Run from the repository root with the local environment configured:

```powershell
.\.venv\Scripts\python.exe -m eval.dataset eval/test_set
.\.venv\Scripts\python.exe -m eval.audit_dataset
```

The first command checks JSONL structure and review metadata without generation.
The current dataset validator reports 75 valid records and 0/75 independently
verified; dataset validity is not proof of completed independent review.
The corpus audit needs the real indexed database and checks supporting evidence
and recorded absence checks. It does not replace independent human review.
Use [Evaluation review](evaluation-review.md) for that process.

Do not rebuild the canonical test files with `eval.build_test_set --force` after
review begins; rebuilding can replace corrections and review metadata.

## Run the comparison experiment

`eval/run_experiment.py` compares the always-answer, fixed-retrieval-threshold,
and calibrated variants. Its generation stage makes live API calls and writes
`eval/raw_run_outputs.jsonl`. This stage opens that file for replacement; archive
an existing run and record the current configuration before starting another.
Only the calibration-seed collector is resumable; the full comparison runner
does not currently implement checkpoint resumption.

```powershell
.\.venv\Scripts\python.exe -m eval.run_experiment --stage generate
```

Review each variant against evidence and ground truth. Add `was_hallucination`
for non-refused responses and save the reviewed records as
`eval/judged_outputs.jsonl`. These are experiment outputs, not training labels.
The script does not automatically judge correctness. Preserve raw outputs and
review decisions together when archiving a research run.

```powershell
.\.venv\Scripts\python.exe -m eval.run_experiment --stage report
```

The generated Markdown table is written to `docs/generated/results_table.md`,
outside the maintained guides and ignored by Git. It reports coverage and
answered/hedged hallucination rates with bootstrap 95% intervals. Coverage is
the fraction with mode other than `refuse`; hallucination rate is measured
among non-refused responses. Inspect category/variant sample counts alongside
each reported rate, especially when few answers are produced.

## Reproducibility and limits

Record corpus source URLs/fiscal periods, embedding and generation model IDs,
model artifact metadata, `TOP_K`, sample count, worker count, thresholds, and
review provenance with each run. Choose thresholds using development validation
only, then freeze them before final evaluation.

The current classifier was trained with an earlier generation model. Revalidate
after changing generation configuration before comparing scientific results.
Unit tests and a successful live web answer do not establish final benchmark
performance, independent review completion, or reduced hallucination rates.
