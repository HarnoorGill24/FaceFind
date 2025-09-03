import sys
from types import SimpleNamespace

# Stub heavy dependencies before importing main
sys.modules["torch"] = SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: False),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
)
sys.modules["facenet_pytorch"] = SimpleNamespace(MTCNN=object)

import main  # after stubbing


def test_iter_media_filters_images_videos(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "c.mp4").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.PNG").touch()
    (sub / "e.mov").touch()
    (sub / "f.doc").touch()

    expected = {
        tmp_path / "a.jpg",
        tmp_path / "c.mp4",
        sub / "d.PNG",
        sub / "e.mov",
    }
    result = set(main.iter_media(tmp_path))
    assert result == expected
