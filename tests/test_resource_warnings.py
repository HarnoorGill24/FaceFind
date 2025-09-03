import sys
import warnings
from types import SimpleNamespace

from PIL import Image

# Stub heavy dependencies before importing modules under test
sys.modules["torch"] = SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: False),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
)

class DummyMTCNN:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, img):
        return [[0, 0, 1, 1]], [0.9]

sys.modules["facenet_pytorch"] = SimpleNamespace(MTCNN=DummyMTCNN)
sys.modules["facefind.quality"] = SimpleNamespace(
    passes_quality=lambda *args, **kwargs: (True, 0.0, "good")
)

import facefind.main as main
import facefind.verify_crops as verify_crops

def _no_resource_warnings(warns):
    return [w for w in warns if issubclass(w.category, ResourceWarning)]

def test_read_image_pil_rgb_no_resource_warning(tmp_path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4)).save(img_path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main.read_image_pil_rgb(img_path)

    assert not _no_resource_warnings(w)

def test_verify_crops_no_resource_warning(tmp_path, monkeypatch):
    img_path = tmp_path / "crop.jpg"
    Image.new("RGB", (4, 4)).save(img_path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        monkeypatch.setattr(
            sys, "argv", ["verify_crops", "--input", str(tmp_path)]
        )
        verify_crops.main()

    assert not _no_resource_warnings(w)
