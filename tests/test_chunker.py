from app.ingestion.chunker import chunk_document, split_into_sections

SAMPLE = (
    "Company overview text. " * 30
    + "\nItem 7. Management's Discussion and Analysis of Financial Condition\n"
    + "revenue grew steadily " * 170          # 510 words -> forces secondary split
    + "\nItem 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
    + "interest rate exposure " * 60
    + "\nItem 8. Financial Statements and Supplementary Data\n"
    + "assets liabilities equity " * 60
)


def test_preamble_before_first_header_is_retained():
    sections = split_into_sections(SAMPLE)
    assert sections[0][0] == "unlabeled"
    assert "Company overview" in sections[0][1]


def test_item_7a_is_its_own_section_not_merged_into_mda():
    sections = split_into_sections(SAMPLE)
    labels = [label for label, _ in sections]
    assert "market_risk" in labels
    mda_text = dict(sections)["MD&A"]
    assert "interest rate exposure" not in mda_text


def test_chunks_never_cross_section_boundaries_and_respect_cap():
    chunks = chunk_document(SAMPLE)
    by_label = {}
    for c in chunks:
        by_label.setdefault(c.section, []).append(c)
    assert set(by_label) == {"unlabeled", "MD&A", "market_risk", "financial_statements"}
    assert all(len(c.text.split()) <= 400 for c in chunks)
    assert any(c.section == "MD&A" for c in chunks)
    # secondary split actually happened inside MD&A
    assert len(by_label["MD&A"]) >= 2


def test_document_without_headers_is_single_unlabeled_chunk():
    chunks = chunk_document("Just some plain text without any section headers. " * 10)
    assert len(chunks) == 1
    assert chunks[0].section == "unlabeled"


def test_empty_and_whitespace_inputs_produce_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []
