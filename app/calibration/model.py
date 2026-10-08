"""
The core novel mechanism (Section 8 of the design doc): a small classifier
mapping (retrieval_confidence, spread, self_consistency_score) -> P(correct),
which is then thresholded into answer / hedge / refuse.

Deliberately simple (logistic regression, with an MLP swap-in option) —
the contribution is signal design + evaluation, not model complexity.
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from app.calibration.features import FEATURE_NAMES

MODEL_DIR = Path("models")


@dataclass
class Thresholds:
    # P(correct) >= answer_threshold      -> "answer"
    # hedge_threshold <= P < answer_thr    -> "hedge"
    # P < hedge_threshold                  -> "refuse"
    answer_threshold: float = 0.70
    hedge_threshold: float = 0.40


class CalibrationModel:
    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        self.clf = (
            LogisticRegression(class_weight="balanced")
            if kind == "logreg"
            else MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000)
        )
        self.thresholds = Thresholds()
        self.version = "v0"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]  # P(correct)

    def decide(self, prob_correct: float) -> str:
        if prob_correct >= self.thresholds.answer_threshold:
            return "answer"
        if prob_correct >= self.thresholds.hedge_threshold:
            return "hedge"
        return "refuse"

    def save(self, name: str = "calibration_model"):
        import joblib
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.clf, MODEL_DIR / f"{name}_{self.version}.joblib")
        meta = {
            "kind": self.kind,
            "version": self.version,
            "features": FEATURE_NAMES,
            "thresholds": asdict(self.thresholds),
        }
        (MODEL_DIR / f"{name}_{self.version}.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, name: str = "calibration_model", version: str = "v0"):
        import joblib
        meta = json.loads((MODEL_DIR / f"{name}_{version}.json").read_text())
        instance = cls(kind=meta["kind"])
        instance.clf = joblib.load(MODEL_DIR / f"{name}_{version}.joblib")
        instance.thresholds = Thresholds(**meta["thresholds"])
        instance.version = meta["version"]
        return instance
