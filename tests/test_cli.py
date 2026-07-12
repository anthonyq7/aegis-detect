import json

import pytest
import torch

from aegis.cli import build_parser, get_code_input, main
from aegis.predictor import DEFAULT_THRESHOLD, Predictor, validate_threshold


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kwargs):
        return _FakeOutput(self._logits)


class _FakeTokenizer:
    def __call__(self, code, **kwargs):
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}


# build_parser()


def test_text_and_file_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--text", "x", "--file", "y"])
    assert exc.value.code == 2


def test_input_is_required():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_json_flag_and_threshold_parsing():
    args = build_parser().parse_args(["--text", "x", "--json", "--threshold", "0.7"])
    assert args.json is True
    assert args.threshold == 0.7
    assert isinstance(args.threshold, float)


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


# get_code_input()


def test_get_code_input_reads_file(tmp_path):
    file = tmp_path / "snippet.py"
    file.write_text("print('hi')", encoding="utf-8")
    assert get_code_input(None, str(file)) == "print('hi')"


def test_get_code_input_missing_file():
    with pytest.raises(SystemExit):
        get_code_input(None, "/does/not/exist.py")


def test_get_code_input_empty_text_raises():
    # regression: --text "" was falsy and fell through to Path(None)
    with pytest.raises(SystemExit):
        get_code_input("", None)


# validate_threshold()


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_validate_threshold_out_of_range(bad):
    with pytest.raises(ValueError):
        validate_threshold(bad)


@pytest.mark.parametrize("ok", [0.0, 1.0])
def test_validate_threshold_bounds_accepted(ok):
    assert validate_threshold(ok) == ok


def test_validate_threshold_none_returns_default():
    assert validate_threshold(None) == DEFAULT_THRESHOLD == 0.7


# predict()


def test_predict_keys_and_ai_equals_threshold_is_ai():
    predictor = Predictor.__new__(Predictor)
    predictor.threshold = 0.5
    predictor.tokenizer = _FakeTokenizer()
    predictor.model = _FakeModel(torch.tensor([[0.0, 0.0]]))

    result = predictor.predict("some code")

    assert set(result) == {"human", "ai", "prediction"}
    assert result["ai"] == "0.5000"
    # ai == threshold must classify as ai-generated (>=, not >)
    assert result["prediction"] == "ai-generated"


# main()


def test_main_json_round_trips(fake_predictor, capsys):
    main(["--text", "print(1)", "--json"])
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data == {"human": "0.2000", "ai": "0.8000", "prediction": "ai-generated"}


def test_main_human_readable(fake_predictor, capsys):
    main(["--text", "print(1)"])
    out = capsys.readouterr().out
    assert "Prediction: ai-generated" in out
    assert "human=0.2000" in out
    assert "ai=0.8000" in out
