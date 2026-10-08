"""
End-to-end sanity check: reads a downloaded filing's text, chunks it
section-aware, embeds each chunk, inserts into pgvector.

Assumes app/ingestion/edgar_fetch.py has already been run and produced
plain text under data/raw_filings/. Adjust path parsing to match whatever
sec-edgar-downloader's actual output layout looks like once you run it —
this is a starting skeleton, not a final parser.
"""
import argparse
from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.retrieval.embed import embed_batch
from db.connection import get_connection


def insert_document(conn, company: str, doc_type: str, fiscal_period: str, source_url: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (company, doc_type, fiscal_period, source_url)
               VALUES (%s, %s, %s, %s) RETURNING doc_id;""",
            (company, doc_type, fiscal_period, source_url),
        )
        doc_id = cur.fetchone()["doc_id"]
    conn.commit()
    return doc_id


def ingest_file(path: Path, company: str, doc_type: str, fiscal_period: str):
    text = path.read_text(errors="ignore")
    chunks = chunk_document(text)
    if not chunks:
        print(f"No chunks produced for {path}, skipping.")
        return

    embeddings = embed_batch([c.text for c in chunks])

    conn = get_connection()
    try:
        doc_id = insert_document(conn, company, doc_type, fiscal_period, str(path))
        with conn.cursor() as cur:
            for chunk, vec in zip(chunks, embeddings):
                cur.execute(
                    """INSERT INTO chunks (doc_id, section, text, embedding, page_ref)
                       VALUES (%s, %s, %s, %s, %s);""",
                    (doc_id, chunk.section, chunk.text, vec, chunk.page_ref),
                )
        conn.commit()
        print(f"Ingested {len(chunks)} chunks from {path} (doc_id={doc_id})")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a plain-text filing")
    parser.add_argument("--company", required=True)
    parser.add_argument("--doc_type", default="10-K")
    parser.add_argument("--fiscal_period", default="unknown")
    args = parser.parse_args()

    ingest_file(Path(args.file), args.company, args.doc_type, args.fiscal_period)
