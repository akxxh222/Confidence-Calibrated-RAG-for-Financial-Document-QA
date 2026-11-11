# Calibration Seed Protocol

## Current artifacts and configuration

The reviewed development dataset is [eval/labeled_seed.jsonl](../eval/labeled_seed.jsonl).
The classifier is [models/calibration_model_v0.joblib](../models/calibration_model_v0.joblib);
its [JSON metadata](../models/calibration_model_v0.json) records feature order,
thresholds, training data hash, and validation results. The joblib file is a
small correctness classifier; Gemini's generation weights are hosted by Google.

The seed has 42 rows: 23 labeled correct and 19 labeled incorrect. The saved
model uses a stratified 29-row training split and 13-row validation split.
Its stored answer threshold is 0.50 and hedge threshold is 0.20. These values
override the fallback defaults in the Python class when the artifact loads.

The feature vector is `[top1_score, spread, self_consistency_score]`. Logistic
regression estimates `p = 1 / (1 + exp(-(w dot x + b)))`. Routing is:

- `p >= 0.50`: answer;
- `0.20 <= p < 0.50`: hedge, adding a qualification to the generated answer;
- `p < 0.20`: refuse.

`app/calibration/train.py` searches candidate threshold pairs on development
validation data. It scores each pair with
`0.65 * answer_precision + 0.25 * answer_coverage + 0.10 * routing_balanced_accuracy`.
Answer candidates start at 0.50 and advance by 0.05; hedge candidates start at
0.20 and remain below the answer threshold. The final evaluation set is never
used for fitting or threshold selection.

The saved validation AUROC is 1.0 and Brier score is about 0.239, on only 13
validation rows. Mean probabilities are about 0.513 for correct and 0.490 for
incorrect rows, so these results do not establish strong probability calibration.
The seed was collected with `gemini-3.8-flash`; runtime now uses
`gemini-3.5-flash`. Recollect and review development generations, then revalidate
the classifier before treating its probabilities as validated for the new model.
Do not alter historical model provenance to match the runtime configuration.

All commands below run from the repository root, using the local Python
environment described in [Getting started](getting-started.md).

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
