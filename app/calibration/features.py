"""
Builds the feature vector fed into the calibration model.

Feature order is fixed and versioned (see model.py FEATURE_NAMES) so that a
saved model always maps predictably onto retrain/eval data.
"""
from dataclasses import dataclass


@dataclass
class CalibrationFeatures:
    top1_score: float
    spread: float               # top1 - top3 retrieval similarity
    self_consistency_score: float

    def as_vector(self) -> list[float]:
        return [self.top1_score, self.spread, self.self_consistency_score]


FEATURE_NAMES = ["top1_score", "spread", "self_consistency_score"]
