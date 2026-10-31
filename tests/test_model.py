import keras
import numpy as np

from wbc_multistream.model import build_model, compile_model


def test_parameter_count_matches_the_notebook():
    # model.summary() in the notebook: 451,077 trainable + 256 non-trainable.
    model = build_model()
    assert model.count_params() == 451_333
    assert sum(int(np.prod(w.shape)) for w in model.trainable_weights) == 451_077


def test_three_inputs_and_probability_output():
    model = build_model(input_shape=(32, 32, 3))
    assert [i.name for i in model.inputs] == ["raw_input", "od_input", "diffused_input"]
    x = [np.random.rand(3, 32, 32, 3).astype("float32") for _ in range(3)]
    out = model.predict(x, verbose=0)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_save_and_load_round_trip(tmp_path):
    model = compile_model(build_model(input_shape=(32, 32, 3), num_classes=3))
    x = [np.random.rand(2, 32, 32, 3).astype("float32") for _ in range(3)]
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-5)
