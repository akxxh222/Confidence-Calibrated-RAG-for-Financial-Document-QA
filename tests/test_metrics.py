from eval.metrics import compute_category_metrics


def test_all_refused_gives_zero_coverage():
    results = [{"mode": "refuse", "was_hallucination": None} for _ in range(10)]
    m = compute_category_metrics(results)
    assert m["coverage"][0] == 0.0


def test_all_answered_correct_gives_zero_hallucination():
    results = [{"mode": "answer", "was_hallucination": False} for _ in range(10)]
    m = compute_category_metrics(results)
    assert m["coverage"][0] == 1.0
    assert m["hallucination_rate"][0] == 0.0


def test_mixed_results():
    results = (
        [{"mode": "answer", "was_hallucination": True} for _ in range(3)]
        + [{"mode": "answer", "was_hallucination": False} for _ in range(7)]
    )
    m = compute_category_metrics(results)
    assert abs(m["hallucination_rate"][0] - 0.3) < 1e-9
