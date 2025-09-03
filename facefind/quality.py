from __future__ import annotations

import numpy as np
from PIL import Image

try:  # pragma: no cover - exercised in tests
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


def _require_cv2() -> None:
    """Raise an informative error if OpenCV is missing."""
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for image quality assessment. "
            "Install the 'opencv-python' package to enable this feature."
        )

def variance_of_laplacian(
    pil: Image.Image, box: tuple[int, int, int, int] | None = None
) -> float:
    """Return variance of Laplacian; crop to *box* if provided."""
    _require_cv2()
    if box is not None:
        pil = pil.crop(box)
    arr = np.array(pil)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def check_exposure(
    pil: Image.Image,
    dark_threshold: int = 50,
    bright_threshold: int = 205,
    tol: float = 0.05,
) -> str:
    """Classify exposure as 'under', 'over', or 'good'."""
    _require_cv2()
    arr = np.array(pil)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total = gray.size
    dark_ratio = float(hist[:dark_threshold].sum()) / total
    bright_ratio = float(hist[bright_threshold:].sum()) / total
    if dark_ratio > tol:
        return "under"
    if bright_ratio > tol:
        return "over"
    return "good"

def passes_quality(
    pil: Image.Image,
    min_var: float = 100.0,
    exposure_tol: float = 0.05,
) -> tuple[bool, float, str]:
    """Return (passes, var, exposure) for convenience."""
    _require_cv2()
    var = variance_of_laplacian(pil)
    exposure = check_exposure(pil, tol=exposure_tol)
    return var >= min_var and exposure == "good", var, exposure
