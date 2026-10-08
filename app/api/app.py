"""
Flask REST API. Single endpoint: POST /ask

Request:  {"question": "..."}
Response: AskResponse (see schemas.py)

This orchestrates the full pipeline described in Section 6 of the design doc:
retrieval -> generation (N samples) -> self-consistency score -> calibration
model -> answer/hedge/refuse decision.
"""
from flask import Flask, request, jsonify

from app.retrieval.retriever import retrieve
from app.generation.self_consistency import run_self_consistency
from app.calibration.model import CalibrationModel
from app.calibration.features import CalibrationFeatures
from app.api.schemas import AskResponse

app = Flask(__name__)

_calibration_model = None


def get_calibration_model() -> CalibrationModel:
    global _calibration_model
    if _calibration_model is None:
        _calibration_model = CalibrationModel.load()
    return _calibration_model


HEDGE_PREFIX = (
    "I'm not fully confident in this answer based on the retrieved filings — "
    "please verify against the source. "
)
REFUSE_MESSAGE = (
    "I don't have sufficient confidence in the retrieved context to answer this "
    "reliably. This may mean the information isn't in the indexed filings, or "
    "the question's premise doesn't match what the filings show."
)


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    retrieval = retrieve(question)
    if not retrieval.chunks:
        resp = AskResponse(
            question=question, mode="refuse", answer=REFUSE_MESSAGE,
            confidence_score=0.0, retrieval_confidence=0.0,
            self_consistency_score=0.0, sources=[],
        )
        return jsonify(resp.to_dict())

    sc_result = run_self_consistency(question, retrieval.chunks)

    features = CalibrationFeatures(
        top1_score=retrieval.top1_score,
        spread=retrieval.spread,
        self_consistency_score=sc_result.self_consistency_score,
    )

    model = get_calibration_model()
    prob_correct = float(model.predict_proba([features.as_vector()])[0])
    mode = model.decide(prob_correct)

    # Use the first sample as the primary answer text; all N samples are
    # available in sc_result.samples if you want to log/inspect disagreement.
    primary_answer = sc_result.samples[0]

    if mode == "refuse":
        answer_text = REFUSE_MESSAGE
    elif mode == "hedge":
        answer_text = HEDGE_PREFIX + primary_answer
    else:
        answer_text = primary_answer

    sources = [
        {"document_id": c.doc_id, "section": c.section, "page_ref": c.page_ref}
        for c in retrieval.chunks
    ]

    resp = AskResponse(
        question=question,
        mode=mode,
        answer=answer_text,
        confidence_score=round(prob_correct, 3),
        retrieval_confidence=round(retrieval.top1_score, 3),
        self_consistency_score=round(sc_result.self_consistency_score, 3),
        sources=sources,
    )
    return jsonify(resp.to_dict())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
