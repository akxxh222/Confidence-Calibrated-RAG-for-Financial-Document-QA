"""
Trains the calibration model on labeled (features -> was_correct) rows.

Input: a CSV/JSONL with columns matching FEATURE_NAMES plus `label`
(1 = answer was correct when checked against ground truth, 0 = not).
This comes from hand-verifying a sample of generated answers per
Section 8/10 of the design doc — there is no way to skip this step.

Usage:
    python -m app.calibration.train --data eval/labeled_seed.jsonl --kind logreg
"""
import argparse
import json
import numpy as np
from sklearn.model_selection import train_test_split

from app.calibration.model import CalibrationModel, Thresholds
from app.calibration.features import FEATURE_NAMES


def load_labeled_data(path: str):
    X, y = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            X.append([row[name] for name in FEATURE_NAMES])
            y.append(int(row["label"]))
    return np.array(X), np.array(y)


def tune_thresholds(model: CalibrationModel, X_val, y_val) -> Thresholds:
    """
    Sweeps threshold pairs on the validation split to pick the pair that
    minimizes hallucination rate on 'answer'-mode predictions while keeping
    coverage reasonable. This is a starting heuristic — Section 12's full
    experiment is where thresholds get properly validated against the
    3-category eval set, not just this seed split.
    """
    probs = model.predict_proba(X_val)
    best = Thresholds()
    best_score = -1.0
    for answer_thr in np.arange(0.5, 0.95, 0.05):
        for hedge_thr in np.arange(0.2, answer_thr, 0.05):
            answer_mask = probs >= answer_thr
            if answer_mask.sum() == 0:
                continue
            precision_on_answer = y_val[answer_mask].mean()
            coverage = answer_mask.mean()
            # simple joint objective: precision matters more than coverage here
            score = 0.7 * precision_on_answer + 0.3 * coverage
            if score > best_score:
                best_score = score
                best = Thresholds(
                    answer_threshold=float(answer_thr),
                    hedge_threshold=float(hedge_thr),
                )
    return best


def main(data_path: str, kind: str):
    X, y = load_labeled_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = CalibrationModel(kind=kind)
    model.fit(X_train, y_train)
    model.thresholds = tune_thresholds(model, X_val, y_val)
    model.save()

    val_probs = model.predict_proba(X_val)
    print(f"Trained {kind} calibration model on {len(X_train)} rows, "
          f"validated on {len(X_val)}.")
    print(f"Chosen thresholds: {model.thresholds}")
    print(f"Val AUROC-ish check (mean prob for label=1 vs 0): "
          f"{val_probs[y_val == 1].mean():.3f} vs {val_probs[y_val == 0].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--kind", choices=["logreg", "mlp"], default="logreg")
    args = parser.parse_args()
    main(args.data, args.kind)
