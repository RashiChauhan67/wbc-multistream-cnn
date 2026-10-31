import json

import numpy as np
import pytest

from wbc_multistream.data import Split
from wbc_multistream.evaluation import evaluate_model, input_ablation
from wbc_multistream.model import build_model, compile_model
from wbc_multistream.training import TrainConfig, train_and_evaluate

SIZE = (32, 32, 3)
CLASSES = ("c0", "c1", "c2")


@pytest.fixture
def toy_split():
    rng = np.random.default_rng(0)
    views = [rng.random((48, *SIZE), dtype=np.float32) for _ in range(3)]
    y = np.eye(3, dtype=np.float32)[np.repeat(np.arange(3), 16)]
    idx = rng.permutation(48)
    train, test = idx[:36], idx[36:]
    return Split([v[train] for v in views], [v[test] for v in views], y[train], y[test])


def _model():
    return compile_model(build_model(input_shape=SIZE, num_classes=3))


def test_evaluate_model_fields(toy_split):
    result = evaluate_model(_model(), toy_split.x_test, toy_split.y_test, CLASSES, batch_size=4)
    assert result["num_samples"] == 12
    assert np.array(result["confusion_matrix"]).sum() == 12
    assert set(CLASSES) <= set(result["classification_report"])


def test_input_ablation_reports_four_accuracies(toy_split):
    acc = input_ablation(_model(), toy_split.x_test, toy_split.y_test, batch_size=4)
    assert set(acc) == {"full", "without_od", "without_diffused", "raw_only"}
    assert all(0.0 <= a <= 1.0 for a in acc.values())


@pytest.mark.parametrize("protocol", ["notebook", "holdout"])
def test_train_and_evaluate_runs_and_writes_outputs(toy_split, tmp_path, protocol):
    cfg = TrainConfig(epochs=1, batch_size=8, protocol=protocol, val_fraction=0.25)
    result = train_and_evaluate(_model(), toy_split, cfg, CLASSES, tmp_path)
    assert result["protocol"] == protocol and result["epochs_run"] == 1
    assert (tmp_path / "model.keras").exists() and (tmp_path / "history.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["protocol"] == protocol


def test_unknown_protocol_raises(toy_split):
    with pytest.raises(ValueError):
        train_and_evaluate(_model(), toy_split, TrainConfig(protocol="bogus"), CLASSES)
