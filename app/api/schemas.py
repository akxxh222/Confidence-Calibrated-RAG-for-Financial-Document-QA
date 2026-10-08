from dataclasses import dataclass, asdict


@dataclass
class AskResponse:
    question: str
    mode: str                       # 'answer' | 'hedge' | 'refuse'
    answer: str
    confidence_score: float         # P(correct) from calibration model
    retrieval_confidence: float
    self_consistency_score: float
    sources: list[dict]

    def to_dict(self):
        return asdict(self)
