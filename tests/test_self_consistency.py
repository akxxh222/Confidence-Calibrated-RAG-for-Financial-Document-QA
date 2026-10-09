from app.generation.self_consistency import _extract_number, _numeric_agreement


def test_extract_plain_number():
    assert _extract_number("Revenue was 383.285 billion.") == 383.285e9


def test_extract_percentage():
    assert _extract_number("Growth was 4%") == 4.0


def test_extract_skips_year_before_financial_answer():
    assert _extract_number("In 2025, revenue was $1.2 billion.") == 1.2e9


def test_extract_normalizes_equivalent_million_and_billion_scales():
    billion = _extract_number("Revenue was $1.2 billion.")
    million = _extract_number("Revenue was $1,200 million.")
    assert billion == million == 1.2e9


def test_extract_parenthesized_financial_value_as_negative():
    assert _extract_number("Operating loss was ($5.2 million).") == -5.2e6


def test_extract_range_uses_first_bound_consistently():
    assert _extract_number("Gross margin was between 10% and 12%.") == 10.0


def test_extract_skips_multiple_years_before_first_financial_figure():
    text = "In 2025 and 2024, revenue was $1.2 billion versus $1.1 billion."
    assert _extract_number(text) == 1.2e9


def test_extract_prioritizes_scaled_figure_after_full_date():
    text = (
        "Microsoft's total revenue for the year ended June 30, 2026 "
        "was $331,839 million."
    )
    assert _extract_number(text) == 331839e6


def test_extract_none_when_no_number():
    assert _extract_number("Revenue grew significantly year over year.") is None


def test_numeric_agreement_all_same():
    assert _numeric_agreement([100.0, 100.0, 100.5]) == 1.0


def test_numeric_agreement_disagreement():
    score = _numeric_agreement([100.0, 50.0])
    assert score == 0.0


def test_numeric_agreement_insufficient_data_returns_none():
    assert _numeric_agreement([None, None]) is None
    assert _numeric_agreement([100.0]) is None
