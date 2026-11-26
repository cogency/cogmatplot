"""Convert arrays into the BGR uint8 images OpenCV shows."""

from __future__ import annotations

from typing import Optional

import numpy as np


def prepare_bgr(image, normalize: bool = True, colormap: Optional[int] = None, rgb: bool = False):
    """Return a BGR ``uint8`` image, or ``None`` when the array cannot be shown."""
    import cv2

    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        if hasattr(image, "__array__"):
            image = np.asarray(image)
        else:
            return None
    if image.size == 0:
        return None

    im = image
    if im.ndim == 2:
        im = _to_uint8(im, normalize)
        if colormap is not None:
            return cv2.applyColorMap(im, colormap)
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    if im.ndim == 3 and im.shape[2] == 3:
        im = _to_uint8(im, normalize)
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if rgb else im

    if im.ndim == 3 and im.shape[2] == 4:
        bgr = im[:, :, :3]
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR) if rgb else bgr

    return None


def _to_uint8(im, normalize: bool):
    if im.dtype == np.uint8:
        return im
    if normalize:
        imf = im.astype(np.float64)
        lo = np.nanmin(imf)
        hi = np.nanmax(imf)
        if hi > lo:
            return ((imf - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return np.zeros(im.shape, dtype=np.uint8)
    return np.clip(im, 0, 255).astype(np.uint8)
