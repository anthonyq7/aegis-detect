import pytest


class FakePredictor:
    """Stand-in for Predictor that never touches the network or torch weights."""

    def __init__(self, threshold=None, model_name="anthonyq7/aegis"):
        self.threshold = threshold

    def predict(self, code):
        return {"human": "0.2000", "ai": "0.8000", "prediction": "ai-generated"}


@pytest.fixture
def fake_predictor(monkeypatch):
    monkeypatch.setattr("aegis.cli.Predictor", FakePredictor)
    return FakePredictor
