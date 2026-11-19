"""Image validation and resizing utilities for ELPV solar cell preprocessing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def validate_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalize an input image array into a 2D uint8 grayscale image."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("Input image is empty")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError("Input floating-point image contains NaN or Inf values")

    arr = image
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            return arr[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Validate grayscale image and resize to target_size=(height, width) using INTER_AREA."""
    img_gray = validate_grayscale_image(image)
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_size dimensions must be positive, got {target_size}")

    if img_gray.shape == (target_h, target_w):
        return img_gray.copy()

    return cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)


def preprocess_for_features(
    image: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Produce standardized uint8 grayscale images for HOG and LBP extraction."""
    prep_cfg = cfg["preprocessing"]
    hog_size = tuple(int(v) for v in prep_cfg["hog_resize"])
    lbp_size = tuple(int(v) for v in prep_cfg["lbp_resize"])

    img_hog = resize_image(image, target_size=hog_size)
    if lbp_size == hog_size:
        img_lbp = img_hog
    else:
        img_lbp = resize_image(image, target_size=lbp_size)
    return img_hog, img_lbp
