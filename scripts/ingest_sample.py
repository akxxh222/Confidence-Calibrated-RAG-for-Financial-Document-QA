"""
Chunk, embed, and ingest one filing or an extracted EDGAR corpus.

Corpus mode understands sec-edgar-downloader's observed layout:
``sec-edgar-filings/<ticker>/<form>/<accession>/``. It reads the reporting
period and registrant CIK from each matching ``full-submission.txt`` file.
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.retrieval.embed import embed_batch
from db.connection import get_connection


@dataclass(frozen=True)
class FilingMetadata:
    company: str
    doc_type: str
    fiscal_period: str
    source_url: str


def discover_filings(root: Path) -> list[Path]:
    """Find extracted primary documents in sec-edgar-downloader's layout."""
    return sorted(root.rglob("primary-document.txt"))


def parse_edgar_metadata(path: Path, raw_root: Path) -> FilingMetadata:
    """Derive ticker/form/accession and period from a downloaded filing."""
    parts = path.parts
    try:
        marker = parts.index("sec-edgar-filings")
        company, doc_type, accession = parts[marker + 1:marker + 4]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unrecognized EDGAR filing path: {path}") from exc

    submission = raw_root.joinpath(
        "sec-edgar-filings", company, doc_type, accession,
        "full-submission.txt",
    )
    header = submission.read_text(encoding="utf-8", errors="ignore")[:10000]
    period_match = re.search(
        r"^CONFORMED PERIOD OF REPORT:\s*(\d{8})", header, re.M
    )
    if period_match is None:
        raise ValueError(f"Missing reporting period in {submission}")
    raw_period = period_match.group(1)
    fiscal_period = f"{raw_period[:4]}-{raw_period[4:6]}-{raw_period[6:]}"

    cik_match = re.search(r"^\s*CENTRAL INDEX KEY:\s*(\d+)", header, re.M)
    if cik_match is None:
        raise ValueError(f"Missing central index key in {submission}")
    cik = cik_match.group(1).lstrip("0")
    accession_compact = accession.replace("-", "")
    source_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_compact}/{accession}-index.html"
    )
    return FilingMetadata(company, doc_type, fiscal_period, source_url)


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


def ingest_file(
    path: Path,
    company: str,
    doc_type: str,
    fiscal_period: str,
    source_url: str,
):
    text = path.read_text(errors="ignore")
    chunks = chunk_document(text)
    if not chunks:
        print(f"No chunks produced for {path}, skipping.")
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM documents WHERE source_url = %s LIMIT 1;",
                (source_url,),
            )
            existing = cur.fetchone()
    finally:
        conn.close()
    if existing:
        print(f"Already ingested {path} (doc_id={existing['doc_id']}), skipping.")
        return

    embeddings = embed_batch([c.text for c in chunks])

    conn = get_connection()
    try:
        doc_id = insert_document(conn, company, doc_type, fiscal_period, source_url)
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


def ingest_corpus(clean_root: Path, raw_root: Path):
    filings = discover_filings(clean_root)
    if not filings:
        raise SystemExit(f"No primary-document.txt files found under {clean_root}")
    for path in filings:
        metadata = parse_edgar_metadata(path, raw_root)
        ingest_file(
            path,
            metadata.company,
            metadata.doc_type,
            metadata.fiscal_period,
            metadata.source_url,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Path to one plain-text filing")
    source.add_argument("--root", help="Root containing extracted EDGAR filings")
    parser.add_argument("--raw_root", default="data/raw_filings")
    parser.add_argument("--company")
    parser.add_argument("--doc_type", default="10-K")
    parser.add_argument("--fiscal_period", default="unknown")
    args = parser.parse_args()

    if args.root:
        ingest_corpus(Path(args.root), Path(args.raw_root))
    else:
        if not args.company:
            parser.error("--company is required with --file")
        ingest_file(
            Path(args.file), args.company, args.doc_type, args.fiscal_period,
            str(args.file),
        )
