"""
The two baselines the calibrated system is compared against (Section 12).

always_answer:   never refuses, never hedges — just returns the top generation.
fixed_threshold: single global similarity cutoff on retrieval.top1_score only,
                 no self-consistency signal at all. This is the brittle
                 baseline the whole project argues against.
"""
from app.retrieval.retriever import retrieve
from app.generation.generate import generate_answer

FIXED_THRESHOLD = 0.55  # tune once on a validation split, then freeze


def run_always_answer(question: str) -> dict:
    retrieval = retrieve(question)
    if not retrieval.chunks:
        return {"mode": "refuse", "answer": "No relevant context found.",
                "retrieval_confidence": 0.0}
    answer = generate_answer(question, retrieval.chunks)
    return {"mode": "answer", "answer": answer,
            "retrieval_confidence": retrieval.top1_score}


def run_fixed_threshold(question: str, threshold: float = FIXED_THRESHOLD) -> dict:
    retrieval = retrieve(question)
    if not retrieval.chunks or retrieval.top1_score < threshold:
        return {"mode": "refuse", "answer": "Insufficient retrieval confidence.",
                "retrieval_confidence": retrieval.top1_score if retrieval.chunks else 0.0}
    answer = generate_answer(question, retrieval.chunks)
    return {"mode": "answer", "answer": answer,
            "retrieval_confidence": retrieval.top1_score}
