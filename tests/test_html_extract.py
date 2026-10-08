from pathlib import Path

from app.ingestion.html_extract import extract_file, extract_text

HTML = """
<html>
<head><title>10-K Filing</title><style>.x { color: red; }</style></head>
<body>
<script>var tracking = 1;</script>
<ix:header><ix:hidden>hidden-xbrl-metadata-noise</ix:hidden></ix:header>
<div>
  <p>Revenue for fiscal 2023 was $383,285 million &amp; growing.</p>
  <table>
    <tr><th>Item</th><th>FY2023</th></tr>
    <tr><td>Total revenue</td><td>$383,285</td></tr>
    <tr><td>Net income</td><td>$96,995</td></tr>
  </table>
  <p>See&nbsp;Item&nbsp;8 for details.</p>
</div>
</body>
</html>
"""


def test_strips_script_style_head():
    text = extract_text(HTML)
    assert "tracking" not in text
    assert "color: red" not in text
    assert "10-K Filing" not in text  # <title> lives in stripped <head>


def test_strips_inline_xbrl_noise():
    text = extract_text(HTML)
    assert "hidden-xbrl-metadata-noise" not in text


def test_entities_decoded_and_nbsp_normalized():
    text = extract_text(HTML)
    assert "$383,285 million & growing." in text
    assert "See Item 8 for details." in text


def test_table_rendered_rowwise_with_cell_separator():
    text = extract_text(HTML)
    assert "Item | FY2023" in text
    assert "Total revenue | $383,285" in text
    assert "Net income | $96,995" in text
    # cells must not be jammed together
    assert "Revenue$383,285" not in text


def test_whitespace_is_collapsed():
    text = extract_text(HTML)
    assert "  " not in text
    assert "\n\n" not in text


def test_no_lxml_fallback_needed_on_plain_html():
    # sanity: works when called repeatedly / on minimal documents
    assert extract_text("<p>hello world</p>") == "hello world"
    assert extract_text("") == ""


def test_extract_file_lets_parser_detect_legacy_html_encoding(tmp_path: Path):
    src = tmp_path / "filing.html"
    src.write_bytes(
        b'<html><head><meta charset="windows-1252"></head>'
        b"<body>Management\x92s Discussion</body></html>"
    )

    dest = extract_file(src, tmp_path / "out")

    assert dest.read_text(encoding="utf-8") == "Management’s Discussion"
