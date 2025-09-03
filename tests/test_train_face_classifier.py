import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from facefind import train_face_classifier


def test_train_face_classifier_artifacts_and_cv(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for person in ["alice", "bob"]:
        d = data_dir / person
        d.mkdir(parents=True)
        (d / f"{person}1.jpg").write_bytes(b"1")
        (d / f"{person}2.jpg").write_bytes(b"1")

    out_dir = tmp_path / "models"

    def fake_load_images(paths):
        return [object()] * len(paths)

    def fake_embed_images(imgs, device=None, batch_size=None):
        return np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def fake_get_device(preferred=None):
        return "cpu"

    def fake_cv(clf, X, y, cv=None, scoring=None):
        if isinstance(clf, KNeighborsClassifier):
            return np.array([0.9, 0.9])
        return np.array([0.6, 0.6])

    monkeypatch.setattr(train_face_classifier, "load_images", fake_load_images)
    monkeypatch.setattr(train_face_classifier, "embed_images", fake_embed_images)
    monkeypatch.setattr(train_face_classifier, "get_device", fake_get_device)
    monkeypatch.setattr(train_face_classifier, "cross_val_score", fake_cv)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_face_classifier", "--data", str(data_dir), "--out", str(out_dir)],
    )
    train_face_classifier.main()

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

    model = joblib.load(out_dir / "face_classifier.joblib")
    assert isinstance(model, KNeighborsClassifier)

