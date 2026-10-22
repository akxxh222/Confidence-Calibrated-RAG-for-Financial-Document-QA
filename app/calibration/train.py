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
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from app.calibration.model import CalibrationModel, Thresholds
from app.calibration.features import FEATURE_NAMES


def load_labeled_data(path: str):
    X, y = [], []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                features = [float(row[name]) for name in FEATURE_NAMES]
                label = int(row["label"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid calibration row"
                ) from error
            if not np.isfinite(features).all():
                raise ValueError(f"{path}:{line_number}: features must be finite")
            if not 0.0 <= features[0] <= 1.0:
                raise ValueError(f"{path}:{line_number}: top1_score outside [0, 1]")
            if not 0.0 <= features[1] <= 2.0:
                raise ValueError(f"{path}:{line_number}: spread outside [0, 2]")
            if not 0.0 <= features[2] <= 1.0:
                raise ValueError(
                    f"{path}:{line_number}: self_consistency_score outside [0, 1]"
                )
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
            X.append(features)
            y.append(label)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def validate_training_data(y: np.ndarray) -> None:
    counts = np.bincount(y, minlength=2)
    if len(y) < 30 or counts.min() < 10:
        raise ValueError(
            "Calibration seed requires at least 30 rows and 10 rows per class"
        )


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
            non_refuse = probs >= hedge_thr
            routing_accuracy = balanced_accuracy_score(y_val, non_refuse)
            score = (
                0.65 * precision_on_answer
                + 0.25 * coverage
                + 0.10 * routing_accuracy
            )
            if score > best_score:
                best_score = score
                best = Thresholds(
                    answer_threshold=float(answer_thr),
                    hedge_threshold=float(hedge_thr),
                )
    return best


def main(data_path: str, kind: str):
    X, y = load_labeled_data(data_path)
    validate_training_data(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = CalibrationModel(kind=kind)
    model.fit(X_train, y_train)
    model.thresholds = tune_thresholds(model, X_val, y_val)
    val_probs = model.predict_proba(X_val)
    metadata = {
        "data_path": data_path,
        "data_sha256": hashlib.sha256(
            Path(data_path).read_bytes()
        ).hexdigest(),
        "random_state": 42,
        "row_count": int(len(y)),
        "class_counts": {
            "incorrect": int((y == 0).sum()),
            "correct": int((y == 1).sum()),
        },
        "validation_row_count": int(len(y_val)),
        "validation_mean_probability": {
            "incorrect": float(val_probs[y_val == 0].mean()),
            "correct": float(val_probs[y_val == 1].mean()),
        },
        "validation_auroc": float(roc_auc_score(y_val, val_probs)),
        "validation_brier_score": float(brier_score_loss(y_val, val_probs)),
    }
    model.save(training_metadata=metadata)

    print(f"Trained {kind} calibration model on {len(X_train)} rows, "
          f"validated on {len(X_val)}.")
    print(f"Chosen thresholds: {model.thresholds}")
    print(f"Val AUROC-ish check (mean prob for label=1 vs 0): "
          f"{val_probs[y_val == 1].mean():.3f} vs {val_probs[y_val == 0].mean():.3f}")
    print(f"Validation AUROC: {metadata['validation_auroc']:.3f}")
    print(f"Validation Brier score: {metadata['validation_brier_score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--kind", choices=["logreg", "mlp"], default="logreg")
    args = parser.parse_args()
    main(args.data, args.kind)
