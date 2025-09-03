from PIL import Image
import numpy as np

import pytest

cv2 = pytest.importorskip("cv2")

from facefind.quality import variance_of_laplacian, check_exposure

def test_variance_of_laplacian_sharp_vs_blur():
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[40:60, :] = 255
    sharp_img = Image.fromarray(sharp)
    blur_img = Image.fromarray(np.full((100, 100), 127, dtype=np.uint8))
    assert variance_of_laplacian(sharp_img) > variance_of_laplacian(blur_img)

def test_check_exposure_under_over_good():
    dark = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    bright = Image.fromarray(np.full((10, 10), 255, dtype=np.uint8))
    mid = Image.fromarray(np.full((10, 10), 127, dtype=np.uint8))
    assert check_exposure(dark) == "under"
    assert check_exposure(bright) == "over"
    assert check_exposure(mid) == "good"
