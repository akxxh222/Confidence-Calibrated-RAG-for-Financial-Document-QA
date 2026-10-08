from app.calibration.model import CalibrationModel, Thresholds


def make_model(answer_thr=0.7, hedge_thr=0.4) -> CalibrationModel:
    m = CalibrationModel()
    m.thresholds = Thresholds(answer_threshold=answer_thr, hedge_threshold=hedge_thr)
    return m


def test_high_confidence_answers():
    model = make_model()
    assert model.decide(0.9) == "answer"
    assert model.decide(0.7) == "answer"


def test_mid_confidence_hedges():
    model = make_model()
    assert model.decide(0.5) == "hedge"
    assert model.decide(0.4) == "hedge"


def test_low_confidence_refuses():
    model = make_model()
    assert model.decide(0.1) == "refuse"
    assert model.decide(0.39) == "refuse"


def test_boundary_is_inclusive_at_answer_threshold():
    model = make_model(answer_thr=0.7)
    assert model.decide(0.70) == "answer"
    assert model.decide(0.699) == "hedge"
