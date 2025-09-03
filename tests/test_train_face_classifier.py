import json
import subprocess
import sys

import pytest


def test_train_face_classifier_artifacts_and_cv(tmp_path):
    data_dir = tmp_path / "data"
    for person in ["alice", "bob"]:
        d = data_dir / person
        d.mkdir(parents=True)
        (d / f"{person}1.jpg").write_bytes(b"1")
        (d / f"{person}2.jpg").write_bytes(b"1")

    out_dir = tmp_path / "models"

    script = """
import json
import sys
import numpy as np
from facefind import train_face_classifier

def fake_load_images(paths):
    return [object()] * len(paths)

def fake_embed_images(imgs, device=None, batch_size=None):
    return np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)

def fake_get_device(preferred=None):
    return "cpu"

def fake_cv(clf, X, y, cv=None, scoring=None):
    import numpy as np
    if "KNeighborsClassifier" in type(clf).__name__:
        return np.array([0.9, 0.9])
    return np.array([0.6, 0.6])

train_face_classifier.load_images = fake_load_images
train_face_classifier.embed_images = fake_embed_images
train_face_classifier.get_device = fake_get_device
train_face_classifier.cross_val_score = fake_cv
sys.argv = ["train_face_classifier", "--data", "{data}", "--out", "{out}"]
train_face_classifier.main()
""".format(data=data_dir, out=out_dir)

    subprocess.run([sys.executable, "-c", script], check=True)

    required = [
        "face_classifier.joblib",
        "labelmap.json",
        "centroids.json",
        "embeddings.npy",
        "train_paths.json",
        "train_labels.json",
    ]
    for name in required:
        assert (out_dir / name).exists()

    with (out_dir / "labelmap.json").open() as f:
        labelmap = json.load(f)
    assert labelmap == {"alice": 0, "bob": 1}

    with (out_dir / "centroids.json").open() as f:
        centroids = json.load(f)
    assert centroids["alice"] == pytest.approx([1.0, 0.0])
    assert centroids["bob"] == pytest.approx([0.0, 1.0])

    with (out_dir / "face_classifier.joblib").open("rb") as f:
        data = f.read()
    assert b"KNeighborsClassifier" in data
