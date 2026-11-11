"""Flask API for normal and confidence-calibrated financial RAG."""
from flask import Flask, jsonify, render_template, request
from google.genai.errors import APIError

from app.api.schemas import AskResponse
from app.calibration.features import CalibrationFeatures
from app.calibration.model import CalibrationModel
from app.generation.generate import generate_answer
from app.generation.self_consistency import run_self_consistency
from app.retrieval.retriever import retrieve


app = Flask(__name__)
_calibration_model = None

HEDGE_PREFIX = (
    "I'm not fully confident in this answer based on the retrieved filings — "
    "please verify against the source. "
)
REFUSE_MESSAGE = (
    "I don't have sufficient confidence in the retrieved context to answer this "
    "reliably. This may mean the information isn't in the indexed filings, or "
    "the question's premise doesn't match what the filings show."
)


def get_calibration_model() -> CalibrationModel:
    global _calibration_model
    if _calibration_model is None:
        _calibration_model = CalibrationModel.load()
    return _calibration_model


def serialize_sources(chunks) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.doc_id,
            "section": chunk.section,
            "page_ref": chunk.page_ref,
            "similarity": round(chunk.similarity, 4),
        }
        for chunk in chunks
    ]


@app.errorhandler(APIError)
def handle_gemini_error(error):
    """Return JSON instead of Flask's HTML error page for provider failures."""
    app.logger.warning("Gemini request failed with status %s", error.code)
    if error.code == 429:
        return jsonify({
            "error": (
                "Gemini's rate or quota limit has been reached. Check usage "
                "for the API key's project in Google AI Studio before retrying."
            )
        }), 429
    if error.code == 504:
        return jsonify({
            "error": (
                "Gemini timed out before completing the request. "
                "Please try again shortly."
            )
        }), 504
    return jsonify({
        "error": (
            "Gemini is temporarily unavailable. Please wait briefly and try "
            "the question again."
        )
    }), 503


@app.route("/", methods=["GET"])
def research_console():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Normal one-generation RAG endpoint retained for API consumers."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "a JSON question is required"}), 400
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    retrieval = retrieve(question)
    if not retrieval.chunks:
        return jsonify({
            "question": question,
            "answer": "No supporting passages were found in the indexed filings.",
            "retrieval_confidence": 0.0,
            "spread": 0.0,
            "sources": [],
        })

    answer = generate_answer(question, retrieval.chunks)
    return jsonify({
        "question": question,
        "answer": answer,
        "retrieval_confidence": round(retrieval.top1_score, 4),
        "spread": round(retrieval.spread, 4),
        "sources": serialize_sources(retrieval.chunks),
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Run retrieval, repeated generation, calibration, and response routing."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "a JSON question is required"}), 400
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    retrieval = retrieve(question)
    if not retrieval.chunks:
        response = AskResponse(
            question=question,
            mode="refuse",
            answer=REFUSE_MESSAGE,
            confidence_score=0.0,
            retrieval_confidence=0.0,
            spread=0.0,
            self_consistency_score=0.0,
            sources=[],
        )
        return jsonify(response.to_dict())

    consistency = run_self_consistency(question, retrieval.chunks)
    features = CalibrationFeatures(
        top1_score=retrieval.top1_score,
        spread=retrieval.spread,
        self_consistency_score=consistency.self_consistency_score,
    )
    model = get_calibration_model()
    probability = float(model.predict_proba([features.as_vector()])[0])
    mode = model.decide(probability)
    primary_answer = consistency.samples[0]

    if mode == "refuse":
        answer = REFUSE_MESSAGE
    elif mode == "hedge":
        answer = HEDGE_PREFIX + primary_answer
    else:
        answer = primary_answer

    response = AskResponse(
        question=question,
        mode=mode,
        answer=answer,
        confidence_score=round(probability, 3),
        retrieval_confidence=round(retrieval.top1_score, 3),
        spread=round(retrieval.spread, 3),
        self_consistency_score=round(consistency.self_consistency_score, 3),
        sources=serialize_sources(retrieval.chunks),
    )
    return jsonify(response.to_dict())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
