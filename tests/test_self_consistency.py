from app.generation.self_consistency import _extract_number, _numeric_agreement


def test_extract_plain_number():
    assert _extract_number("Revenue was 383.285 billion.") == 383.285e9


def test_extract_percentage():
    assert _extract_number("Growth was 4%") == 4.0


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
