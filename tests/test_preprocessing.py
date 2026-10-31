import numpy as np
import pytest

from wbc_multistream.preprocessing import (
    anisotropic_diffusion,
    anisotropic_diffusion_color,
    build_views,
    color_stain_augmentation,
    optical_density,
    shape_constrained_rbc_augmentation,
)


@pytest.fixture
def cell_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)


def test_optical_density_of_a_white_image_is_zero():
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    np.testing.assert_allclose(optical_density(white), 0.0, atol=1e-6)


def test_optical_density_is_larger_for_darker_pixels():
    dark = np.full((4, 4, 3), 10, dtype=np.uint8)
    light = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert optical_density(dark).mean() > optical_density(light).mean()


def test_optical_density_swaps_bgr_to_rgb():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue in BGR
    od = optical_density(img)
    assert od[0, 0, 2] < od[0, 0, 0]  # blue channel (RGB index 2) is brightest, so lowest OD


def test_diffusion_leaves_a_constant_image_unchanged():
    flat = np.full((16, 16), 3.0, dtype=np.float32)
    np.testing.assert_allclose(anisotropic_diffusion(flat), flat, atol=1e-6)


def test_diffusion_smooths_noise_and_preserves_the_mean():
    rng = np.random.default_rng(1)
    noisy = rng.normal(2.0, 0.5, (32, 32)).astype(np.float32)
    out = anisotropic_diffusion(noisy)
    assert out.std() < noisy.std()
    assert out.mean() == pytest.approx(noisy.mean(), abs=1e-3)


def test_diffusion_option_two_and_invalid_option():
    img = np.random.default_rng(2).random((8, 8)).astype(np.float32)
    assert anisotropic_diffusion(img, option=2).shape == (8, 8)
    with pytest.raises(ValueError):
        anisotropic_diffusion(img, option=3)


def test_kappa_20_barely_limits_diffusion_on_optical_density_scale():
    """OD values stay below about 5.5, so conduction coefficients with kappa=20 are all > 0.92."""
    max_gradient = 5.5
    assert np.exp(-((max_gradient / 20.0) ** 2)) > 0.92


def test_color_diffusion_keeps_shape(cell_image):
    od = optical_density(cell_image)
    assert anisotropic_diffusion_color(od, niter=2).shape == od.shape


def test_build_views_shapes_dtypes_and_raw_scale(cell_image):
    raw, od, diffused = build_views(np.stack([cell_image, cell_image]))
    assert raw.shape == od.shape == diffused.shape == (2, 32, 32, 3)
    assert raw.dtype == od.dtype == diffused.dtype == np.float32
    assert raw.max() > 1.0  # the raw view is not rescaled to [0, 1]


def test_augmentations_return_finite_arrays_of_the_same_shape(cell_image):
    od = optical_density(cell_image)
    rng = np.random.default_rng(3)
    stained = color_stain_augmentation(od, rng=rng)
    morphed = shape_constrained_rbc_augmentation(od, rng=rng)
    for out in (stained, morphed):
        assert out.shape == od.shape and np.isfinite(out).all()
