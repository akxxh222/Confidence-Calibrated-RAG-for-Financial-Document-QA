"""
Downloads 10-K / 10-Q filings from SEC EDGAR for a small set of tickers.

Usage:
    python -m app.ingestion.edgar_fetch --tickers AAPL MSFT --forms 10-K --limit 2

SEC fair-access rules require every automated request to declare who is making
it. Set both in .env before running — a generic or example identity gets
requests blocked by SEC:
    EDGAR_NAME   - your name or the project name
    EDGAR_EMAIL  - a real contact email
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

RAW_DIR = Path("data/raw_filings")


def _edgar_identity() -> tuple[str, str]:
    load_dotenv()
    name = os.environ.get("EDGAR_NAME", "").strip()
    email = os.environ.get("EDGAR_EMAIL", "").strip()
    if not name or not email or "example.com" in email.lower():
        sys.exit(
            "EDGAR_NAME / EDGAR_EMAIL are not properly set in .env. SEC "
            "fair-access rules require declaring a real contact with every "
            "automated request; placeholder or example.com addresses get blocked."
        )
    return name, email


def fetch(tickers: list[str], forms: list[str], limit: int = 2):
    name, email = _edgar_identity()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(name, email, RAW_DIR)
    for ticker in tickers:
        for form in forms:
            dl.get(form, ticker, limit=limit, download_details=True)
            print(f"Downloaded {limit}x {form} for {ticker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--forms", nargs="+", default=["10-K"])
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    fetch(args.tickers, args.forms, args.limit)
