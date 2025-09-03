import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import facefind.main as main
from facefind.config import get_profile
from facefind.utils import is_image


def test_is_image_and_is_video_case_insensitive(tmp_path):
    assert is_image(Path("a.JPG"))
    assert is_image(Path("b.png"))
    assert not is_image(Path("c.txt"))
    assert main.is_video(Path("d.MP4"))
    assert main.is_video(Path("e.mov"))
    assert not main.is_video(Path("f.doc"))


def test_crop_and_save_invalid_box_raises(tmp_path):
    from PIL import Image

    im = Image.new("RGB", (10, 10), color=(255, 0, 0))
    # x2 < x1 yields non-positive width
    with pytest.raises(ValueError):
        main.crop_and_save(im, (5, 2, 3, 8), tmp_path, "x", 0)


def test_bgr_to_pil_rgb_raises_without_cv2(monkeypatch):
    monkeypatch.setattr(main, "cv2", None)
    monkeypatch.setattr(main, "np", None)
    with pytest.raises(RuntimeError):
        main.bgr_to_pil_rgb(SimpleNamespace())


def test_create_mtcnn_uses_cpu_on_mps(monkeypatch):
    prof = get_profile("strict")

    captured = {}

    class DummyMTCNN:
        def __init__(self, keep_all, thresholds, min_face_size, device):
            captured.update(
                keep_all=keep_all,
                thresholds=thresholds,
                min_face_size=min_face_size,
                device=device,
            )

    monkeypatch.setitem(sys.modules, "facenet_pytorch", SimpleNamespace(MTCNN=DummyMTCNN))

    _ = main.create_mtcnn(prof, device="mps")
    assert captured["device"] == "cpu"

    captured.clear()
    _ = main.create_mtcnn(prof, device="cuda")
    assert captured["device"] == "cuda"

