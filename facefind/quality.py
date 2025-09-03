"""Image quality heuristics for filtering face crops."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    import numpy as np
    from PIL import Image

_CV2_SENTINEL = object()
_NP_SENTINEL = object()
cv2 = _CV2_SENTINEL  # type: ignore[assignment]
np = _NP_SENTINEL  # type: ignore[assignment]


def _require_cv2():
    """Return the imported ``cv2`` module or raise a helpful error."""
    global cv2
    if cv2 is _CV2_SENTINEL:
        try:  # pragma: no cover - exercised in tests
            cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError as e:  # pragma: no cover - optional dependency
            cv2 = None
            raise ImportError(
                "OpenCV is required for image quality assessment. "
                "Install the 'opencv-python' package to enable this feature."
            ) from e
    if cv2 is None:  # pragma: no cover - when patched to None in tests
        raise ImportError(
            "OpenCV is required for image quality assessment. "
            "Install the 'opencv-python' package to enable this feature."
        )
    return cv2


def _require_numpy():
    """Return the imported ``numpy`` module or raise a helpful error."""
    global np
    if np is _NP_SENTINEL:
        try:  # pragma: no cover - exercised in tests
            np = importlib.import_module("numpy")
        except ModuleNotFoundError as e:  # pragma: no cover - optional dependency
            np = None
            raise ImportError(
                "NumPy is required for image quality assessment. "
                "Install the 'numpy' package to enable this feature."
            ) from e
    if np is None:  # pragma: no cover - when patched to None in tests
        raise ImportError(
            "NumPy is required for image quality assessment. "
            "Install the 'numpy' package to enable this feature."
        )
    return np


def variance_of_laplacian(
    pil: Image.Image, box: tuple[int, int, int, int] | None = None
) -> float:
    """Return variance of Laplacian; crop to *box* if provided."""
    cv2 = _require_cv2()
    np = _require_numpy()
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
    cv2 = _require_cv2()
    np = _require_numpy()
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
    var = variance_of_laplacian(pil)
    exposure = check_exposure(pil, tol=exposure_tol)
    return var >= min_var and exposure == "good", var, exposure
