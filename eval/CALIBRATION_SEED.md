# Calibration Seed Protocol

The calibration seed is development data. It must remain separate from the
75-question final evaluation set in `eval/test_set`.

## Sampling

- Use every question in `eval/calibration_seed_questions.jsonl`.
- The seed contains 42 unique questions: 20 answerable, 10 false-premise, and
  12 unanswerable.
- The automated test verifies that no normalized question text appears in the
  final evaluation set.
- Questions use fiscal 2024 Apple and fiscal 2025 Microsoft facts or disclosure
  requests that are distinct from the final evaluation questions.
- Collect features using the configured production embedding model, generation
  model, top-k value, and five-sample self-consistency path.

Run the resumable collector with:

```powershell
.\.venv\Scripts\python.exe -m eval.collect_calibration_seed
```

The raw run and progress files are ignored by Git. Progress is saved after
retrieval and after every generation sample, so a quota interruption does not
discard completed API work.

## Label definition

`label=1` means the first generated candidate is a correct, substantive answer
that is supported by the retrieved filing context. `label=0` means the
candidate must not be presented as an answer. Consequently:

- an accurate answer to an answerable question is `1`;
- a false-premise response is `1` only when it rejects the premise and gives
  the supported correction;
- an incorrect figure, unsupported inference, or misleading correction is `0`;
- an insufficient-context response is `0`, including when refusal is the
  appropriate behavior for an unanswerable question. This trains the router to
  hedge or refuse rather than treating a consistent refusal as answerable.

For every row, compare the first generated answer with `expected_answer` and
the retrieved chunks. Record the binary label, a short rationale, reviewer,
review date, generation model, embedding model, and sample count in
`eval/labeled_seed.jsonl`. Review all rows manually before training.

## Training and leakage control

The trainer requires at least 30 rows and at least 10 examples from each
binary class. It creates a deterministic, stratified 70/30 development split
using random seed 42. The final 75-question evaluation set is never used for
model fitting or threshold selection.

Train only after the labels pass review:

```powershell
.\.venv\Scripts\python.exe -m app.calibration.train `
  --data eval/labeled_seed.jsonl --kind logreg
```

The saved metadata records the seed SHA-256 hash, row and class counts,
feature order, thresholds, validation probabilities, AUROC, Brier score, and
random seed. The MLP is not trained unless the logistic-regression result has
first been inspected and a documented reason justifies the added complexity.
