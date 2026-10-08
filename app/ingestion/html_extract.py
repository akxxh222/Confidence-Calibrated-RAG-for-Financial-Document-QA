"""
HTML -> plain-text extraction for SEC EDGAR filings.

edgar_fetch.py downloads filings as inline-XBRL HTML documents; the chunker and
embedder need clean text. This module strips markup, drops script/style/XBRL
noise, and renders tables row-wise with " | " cell separators so tabular
figures stay token-separated instead of being jammed into one long string
(naive get_text() on a financial-statements table produces "Revenue383,285...").

Pipeline position:
    edgar_fetch.py -> html_extract.py (this module) -> ingest_sample.py

Usage:
    python -m app.ingestion.html_extract --input data/raw_filings --out data/clean_text

--input may be a single .htm/.html file or a directory (walked recursively, so
sec-edgar-downloader's nested output layout works as-is). Output .txt files
mirror the relative directory structure under --out.

NOTE: header-pattern tuning and table edge cases in real filings should be
checked against the first few extracted documents before ingesting the corpus.
"""
import argparse
from pathlib import Path

from bs4 import BeautifulSoup

# Tags whose entire content is noise for retrieval purposes.
STRIP_TAGS = ["script", "style", "head", "noscript", "ix:header", "ix:hidden"]

# Rendered one text line per <tr>, cells joined with " | ".
ROW_TAGS = ["td", "th"]


def _table_to_lines(table) -> list[str]:
    """Render a <table> as one text line per row, cells separated by ' | '."""
    lines = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(ROW_TAGS)]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def extract_text(html: str) -> str:
    """Convert one EDGAR filing HTML document to clean plain text."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # lxml can reject pathological markup; fall back
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Replace only top-level tables (nested tables are already captured by
    # their parent's row rendering, which avoids duplicated rows).
    top_tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    for table in top_tables:
        table.replace_with(soup.new_string("\n".join(_table_to_lines(table))))

    # get_text("\n") puts a newline between all strings; per-line whitespace
    # normalization (str.split() also collapses \xa0 from &nbsp;) does the rest.
    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(" ".join(line.split()) for line in lines if line.strip())


def extract_file(src: Path, out_dir: Path, root: Path | None = None) -> Path:
    """Extract one file, writing <stem>.txt under out_dir (mirroring layout)."""
    text = extract_text(src.read_text(errors="ignore"))
    if root is not None:
        rel = src.relative_to(root)
        dest = out_dir / rel.with_suffix(".txt")
    else:
        dest = out_dir / (src.stem + ".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="A .htm/.html file, or a directory to walk recursively")
    parser.add_argument("--out", required=True, help="Directory for extracted .txt files")
    args = parser.parse_args()

    src, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.rglob("*")) if p.is_file()] if src.is_dir() else [src]

    count = 0
    for path in files:
        if path.suffix.lower() not in (".htm", ".html"):
            continue
        dest = extract_file(path, out, root=src if src.is_dir() else None)
        words = len(dest.read_text(encoding="utf-8").split())
        print(f"Extracted {words:>7} words -> {dest}")
        count += 1
    print(f"Done: {count} file(s) extracted to {out}")


if __name__ == "__main__":
    main()
