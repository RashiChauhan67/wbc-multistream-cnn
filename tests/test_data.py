import cv2
import numpy as np
import pytest

from wbc_multistream.data import holdout_split, load_images, make_split

CLASSES = ("a", "b", "c")


@pytest.fixture
def image_root(tmp_path):
    rng = np.random.default_rng(0)
    for name in CLASSES:
        (tmp_path / name).mkdir()
        for i in range(10):
            cv2.imwrite(
                str(tmp_path / name / f"{i}.png"), rng.integers(0, 255, (20, 24, 3), dtype=np.uint8)
            )
        (tmp_path / name / "notes.txt").write_text("not an image")
    return tmp_path


def test_load_images_shapes_dtype_and_labels(image_root):
    images, labels = load_images(image_root, CLASSES, image_size=(16, 16))
    assert images.shape == (30, 16, 16, 3) and images.dtype == np.uint8
    assert labels.tolist() == [0] * 10 + [1] * 10 + [2] * 10


def test_missing_class_folder_raises(image_root):
    with pytest.raises(FileNotFoundError):
        load_images(image_root, ("a", "missing"))


def test_make_split_is_stratified_seeded_and_aligned_across_views(image_root):
    images, labels = load_images(image_root, CLASSES, image_size=(16, 16))
    split = make_split(images, labels, num_classes=3, test_size=0.2)
    again = make_split(images, labels, num_classes=3, test_size=0.2)

    assert [len(v) for v in split.x_train] == [24, 24, 24]
    assert [len(v) for v in split.x_test] == [6, 6, 6]
    assert split.y_test.sum(axis=0).tolist() == [2.0, 2.0, 2.0]  # stratified
    np.testing.assert_array_equal(split.y_test, again.y_test)
    # the raw view and the OD view of one sample stay aligned
    assert split.x_test[0].shape == split.x_test[1].shape == split.x_test[2].shape


def test_holdout_split_keeps_test_untouched(image_root):
    images, labels = load_images(image_root, CLASSES, image_size=(16, 16))
    split = make_split(images, labels, num_classes=3, test_size=0.2)
    fit, val = holdout_split(split, val_fraction=0.25)
    assert len(fit.x_train[0]) + len(val.x_train[0]) == len(split.x_train[0])
    assert set(val.y_train.argmax(axis=1)) == {0, 1, 2}
    assert len(fit.x_test[0]) == len(split.x_test[0]) == 6
