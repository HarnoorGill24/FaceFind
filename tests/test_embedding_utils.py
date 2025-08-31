import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub heavy dependencies before importing embedding_utils
sys.modules["torch"] = SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: False),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
)
sys.modules["facenet_pytorch"] = SimpleNamespace(InceptionResnetV1=object)

from PIL import Image

import embedding_utils


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(embedding_utils.torch.cuda, "is_available", lambda: True)
    assert embedding_utils.get_device() == "cuda"


def test_get_device_prefers_mps(monkeypatch):
    monkeypatch.setattr(embedding_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(embedding_utils.torch.backends.mps, "is_available", lambda: True)
    assert embedding_utils.get_device() == "mps"


def test_get_device_default_cpu(monkeypatch):
    monkeypatch.setattr(embedding_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(embedding_utils.torch.backends.mps, "is_available", lambda: False)
    assert embedding_utils.get_device() == "cpu"


def test_batched_chunks():
    result = list(embedding_utils.batched(range(5), 2))
    assert result == [[0, 1], [2, 3], [4]]


def test_load_images(tmp_path):
    good = tmp_path / "img.png"
    Image.new("RGB", (1, 1)).save(good)
    bad = tmp_path / "bad.png"
    imgs = embedding_utils.load_images([good, bad])
    assert imgs[0] is not None and imgs[1] is None
