from __future__ import annotations

import numpy as np
from PIL import Image

import pytest

import facefind.quality as quality

try:  # pragma: no cover - test helper
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@pytest.mark.skipif(cv2 is None, reason="OpenCV not installed")
def test_variance_of_laplacian_sharp_vs_blur() -> None:
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[40:60, :] = 255
    sharp_img = Image.fromarray(sharp)
    blur_img = Image.fromarray(np.full((100, 100), 127, dtype=np.uint8))
    assert quality.variance_of_laplacian(sharp_img) > quality.variance_of_laplacian(blur_img)


@pytest.mark.skipif(cv2 is None, reason="OpenCV not installed")
def test_check_exposure_under_over_good() -> None:
    dark = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    bright = Image.fromarray(np.full((10, 10), 255, dtype=np.uint8))
    mid = Image.fromarray(np.full((10, 10), 127, dtype=np.uint8))
    assert quality.check_exposure(dark) == "under"
    assert quality.check_exposure(bright) == "over"
    assert quality.check_exposure(mid) == "good"


def test_variance_of_laplacian_requires_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "cv2", None)
    with pytest.raises(ImportError, match="OpenCV"):
        quality.variance_of_laplacian(Image.new("L", (1, 1)))


def test_check_exposure_requires_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "cv2", None)
    with pytest.raises(ImportError, match="OpenCV"):
        quality.check_exposure(Image.new("L", (1, 1)))


def test_passes_quality_requires_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "cv2", None)
    with pytest.raises(ImportError, match="OpenCV"):
        quality.passes_quality(Image.new("L", (1, 1)))

