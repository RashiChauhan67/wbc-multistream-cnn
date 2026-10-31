"""Image views used by the three streams, ported from the original notebook.

* ``raw``: the resized BGR image, **not rescaled** (0 to 255), as fed in the notebook.
* ``od``: optical density, ``-log((RGB + 1) / 256)``.
* ``diffused``: Perona-Malik anisotropic diffusion of the OD image, channel by channel.

The colour-stain and red-blood-cell augmentations from the notebook are included for
completeness. They were **not** used to produce the recorded results.
"""

from __future__ import annotations

import cv2
import numpy as np


def optical_density(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an 8-bit BGR image to optical density (float32, same height and width)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb += 1.0  # avoid log(0)
    rgb /= 256.0
    return -np.log(rgb + 1e-8)


def anisotropic_diffusion(
    img: np.ndarray,
    niter: int = 10,
    kappa: float = 20.0,
    gamma: float = 0.2,
    option: int = 1,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion of a 2-D array (periodic borders via ``np.roll``).

    Args:
        img: 2-D float array.
        niter: Number of iterations.
        kappa: Conduction coefficient. Gradients much smaller than ``kappa`` diffuse freely.
        gamma: Update step (below 0.25 for stability in 2-D).
        option: 1 for ``exp(-(g/kappa)^2)``, 2 for ``1 / (1 + (g/kappa)^2)``.
    """
    if option not in (1, 2):
        raise ValueError("option must be 1 or 2")
    out = img.astype(np.float32).copy()
    for _ in range(niter):
        north = np.roll(out, -1, axis=0) - out
        south = np.roll(out, 1, axis=0) - out
        east = np.roll(out, -1, axis=1) - out
        west = np.roll(out, 1, axis=1) - out
        if option == 1:
            c_n, c_s, c_e, c_w = (np.exp(-((d / kappa) ** 2)) for d in (north, south, east, west))
        else:
            c_n, c_s, c_e, c_w = (1 / (1 + (d / kappa) ** 2) for d in (north, south, east, west))
        out += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return out


def anisotropic_diffusion_color(img: np.ndarray, **kwargs) -> np.ndarray:
    """Apply :func:`anisotropic_diffusion` to each channel of an ``(H, W, C)`` image."""
    channels = [anisotropic_diffusion(ch, **kwargs) for ch in cv2.split(img)]
    return cv2.merge(channels)


def build_views(
    images_bgr: np.ndarray, **diffusion_kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three stream inputs from a batch of 8-bit BGR images.

    Returns:
        ``(raw, od, diffused)``, each float32 with shape ``(N, H, W, 3)``.
    """
    raw = images_bgr.astype(np.float32)
    od = np.stack([optical_density(img) for img in images_bgr])
    diffused = np.stack([anisotropic_diffusion_color(img, **diffusion_kwargs) for img in od])
    return raw, od.astype(np.float32), diffused.astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentations (implemented in the notebook; not used for the recorded results)
# ---------------------------------------------------------------------------
def color_stain_augmentation(
    od_image: np.ndarray,
    max_hue_shift: float = 0.05,
    max_saturation_shift: float = 0.4,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    output_in_od: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random hue / saturation / brightness jitter applied in pseudo-RGB, returned as OD or RGB."""
    rng = rng or np.random.default_rng()
    pseudo_rgb = np.exp(-np.clip(od_image, 0, 5))
    rgb_8u = (np.clip(pseudo_rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_8u, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-max_hue_shift, max_hue_shift) * 180) % 180
    hsv[..., 1] = np.clip(
        hsv[..., 1] * (1.0 + rng.uniform(-max_saturation_shift, max_saturation_shift)), 0, 255
    )
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(*brightness_range), 0, 255)

    aug_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    if output_in_od:
        return (-np.log(np.clip(aug_rgb, 1e-8, 1.0))).astype(np.float32)
    return aug_rgb


def shape_constrained_rbc_augmentation(
    od_img: np.ndarray,
    morph_prob: float = 1.0,
    threshold_value: int = 100,
    kernel_min: int = 3,
    kernel_max: int = 8,
    morph_iterations: int = 2,
    od_shift: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly erode or dilate a thresholded red-blood-cell mask and reduce OD inside it."""
    rng = rng or np.random.default_rng()
    if rng.random() > morph_prob:
        return od_img

    od_gray = od_img.mean(axis=-1) if od_img.ndim == 3 else od_img
    pseudo_gray = np.exp(-np.clip(od_gray, 0, 5))
    _, mask = cv2.threshold(
        (pseudo_gray * 255).astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY
    )

    operation = rng.choice(["erode", "dilate"])
    size = int(rng.integers(kernel_min, kernel_max))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for _ in range(morph_iterations):
        mask = cv2.erode(mask, kernel) if operation == "erode" else cv2.dilate(mask, kernel)

    mask_f = mask.astype(np.float32) / 255.0
    if od_img.ndim == 3:
        return od_img * (1 - od_shift * mask_f[..., None])
    return od_img * (1 - od_shift * mask_f)
