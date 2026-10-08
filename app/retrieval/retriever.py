"""
Top-k similarity search against pgvector, plus the retrieval-confidence
signals used downstream by the calibration model (Section 8 of the design doc):

  - top1_score:   cosine similarity of the best-matching chunk
  - spread:       top1_score - top3_score (large gap => one clear winner;
                   small gap => retrieval is ambiguous about which chunk
                   is actually relevant)
"""
from dataclasses import dataclass
import os

from db.connection import get_connection
from app.retrieval.embed import embed_text

TOP_K_DEFAULT = int(os.environ.get("TOP_K", "5"))  # TOP_K in .env; was previously hardcoded


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    section: str
    text: str
    page_ref: str | None
    similarity: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    top1_score: float
    spread: float          # top1 - top3 similarity


def retrieve(query: str, top_k: int = TOP_K_DEFAULT) -> RetrievalResult:
    query_vec = embed_text(query)

    sql = """
        SELECT chunk_id, doc_id, section, text, page_ref,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vec, query_vec, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    chunks = [
        RetrievedChunk(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], section=r["section"],
            text=r["text"], page_ref=r["page_ref"], similarity=float(r["similarity"]),
        )
        for r in rows
    ]

    if not chunks:
        return RetrievalResult(chunks=[], top1_score=0.0, spread=0.0)

    top1 = chunks[0].similarity
    top3 = chunks[2].similarity if len(chunks) >= 3 else chunks[-1].similarity
    return RetrievalResult(chunks=chunks, top1_score=top1, spread=top1 - top3)


# --- Optional cross-encoder reranking stretch goal (Section 5) ---
# from sentence_transformers import CrossEncoder
# _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#
# def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
#     pairs = [(query, c.text) for c in chunks]
#     scores = _reranker.predict(pairs)
#     ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
#     return [c for c, _ in ranked]
