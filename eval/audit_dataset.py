"""Audit evaluation evidence and absence claims against the live corpus."""

from pathlib import Path

from db.connection import get_connection
from eval.dataset import load_and_validate


TEST_SET_DIR = Path("eval/test_set")


def _normalize(text: str) -> str:
    repaired = text.replace("\u00e2\u20ac\u2122", "\u2019")
    return " ".join(repaired.split()).casefold()


def audit_corpus(records: list[dict]) -> list[str]:
    """Return errors for evidence not found or contradicted absence checks."""
    errors: list[str] = []
    with get_connection() as connection:
        with connection.cursor() as cursor:
            for row in records:
                question_id = row["question_id"]
                for evidence in row["evidence"]:
                    cursor.execute(
                        """
                        SELECT c.text
                        FROM chunks c
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE d.source_url = %s
                        """,
                        (evidence["source_url"],),
                    )
                    excerpt = _normalize(evidence["excerpt"])
                    if not any(
                        excerpt in _normalize(chunk["text"])
                        for chunk in cursor.fetchall()
                    ):
                        errors.append(
                            f"{question_id}: evidence excerpt not found at "
                            f"{evidence['source_url']}"
                        )

                absence = row.get("absence_check")
                if not absence:
                    continue
                for term in absence["search_terms"]:
                    cursor.execute(
                        """
                        SELECT COUNT(*) AS match_count
                        FROM chunks c
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE d.company = %s AND c.text ILIKE %s
                        """,
                        (row["company"], f"%{term}%"),
                    )
                    match_count = cursor.fetchone()["match_count"]
                    if match_count:
                        errors.append(
                            f"{question_id}: absence term {term!r} appears in "
                            f"{match_count} company-scoped chunks"
                        )
    return errors


def main() -> None:
    records = load_and_validate(TEST_SET_DIR)
    errors = audit_corpus(records)
    if errors:
        raise ValueError("Corpus audit failed:\n- " + "\n- ".join(errors))

    evidence_count = sum(len(row["evidence"]) for row in records)
    absence_count = sum(bool(row["absence_check"]) for row in records)
    print(f"Verified evidence excerpts: {evidence_count}")
    print(f"Verified company-scoped absence checks: {absence_count}")


if __name__ == "__main__":
    main()
