from pathlib import Path

from scripts.ingest_sample import discover_filings, parse_edgar_metadata


def test_discovers_only_primary_documents_in_real_downloader_layout(tmp_path: Path):
    filing_dir = (
        tmp_path / "sec-edgar-filings" / "AAPL" / "10-K"
        / "0000320193-25-000079"
    )
    filing_dir.mkdir(parents=True)
    primary = filing_dir / "primary-document.txt"
    primary.write_text("filing", encoding="utf-8")
    (filing_dir / "other.txt").write_text("ignore", encoding="utf-8")

    assert discover_filings(tmp_path) == [primary]


def test_parses_metadata_and_period_from_downloader_layout(tmp_path: Path):
    clean = (
        tmp_path / "clean" / "sec-edgar-filings" / "MSFT" / "10-Q"
        / "0001193125-26-191507" / "primary-document.txt"
    )
    clean.parent.mkdir(parents=True)
    clean.write_text("filing", encoding="utf-8")
    raw = (
        tmp_path / "raw" / "sec-edgar-filings" / "MSFT" / "10-Q"
        / "0001193125-26-191507" / "full-submission.txt"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "CONFORMED PERIOD OF REPORT:\t20260331\n"
        "FORM TYPE:\t10-Q\n"
        "CENTRAL INDEX KEY:\t0000789019\n",
        encoding="utf-8",
    )

    metadata = parse_edgar_metadata(clean, tmp_path / "raw")

    assert metadata.company == "MSFT"
    assert metadata.doc_type == "10-Q"
    assert metadata.fiscal_period == "2026-03-31"
    assert metadata.source_url.endswith(
        "/789019/000119312526191507/0001193125-26-191507-index.html"
    )
