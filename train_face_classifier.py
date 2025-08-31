
#!/usr/bin/env python3
"""
FaceFind - train_face_classifier.py
Create embeddings from a labeled folder tree and train a simple classifier.
Select the best model via cross-validation and save:
- models/face_classifier.joblib
- models/labelmap.json
- models/centroids.json
Respects --strictness profile from config.py to set embedding batch size.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from config import get_profile
from embedding_utils import embed_images, get_device, load_images
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def list_images_with_labels(root: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
    """
    Expects structure:
      root/
        person_a/ img1.jpg, img2.jpg, ...
        person_b/ ...
    Returns: (paths, y_ints, inv_labelmap)
    """
    paths, labels = [], []
    classes = sorted([d for d in root.iterdir() if d.is_dir()])
    name_to_int: Dict[str, int] = {d.name: i for i, d in enumerate(classes)}
    inv_map: Dict[int, str] = {i: d.name for d, i in name_to_int.items()}

    for cls_dir in classes:
        cls_idx = name_to_int[cls_dir.name]
        for p in sorted(cls_dir.rglob('*')):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                paths.append(p)
                labels.append(cls_idx)
    return paths, labels, inv_map

def compute_class_centroids(X: np.ndarray, y: List[int]) -> Dict[int, List[float]]:
    centroids: Dict[int, List[float]] = {}
    y_arr = np.asarray(y)
    for cls in np.unique(y_arr):
        m = X[y_arr == cls].mean(axis=0)
        centroids[int(cls)] = m.tolist()
    return centroids

def main():
    parser = argparse.ArgumentParser(description="Train a face classifier from a labeled folder tree")
    parser.add_argument("--data", required=True, help="Path to labeled people folder (each subfolder = class)")
    parser.add_argument("--out", default="models", help="Output directory (default: models)")
    parser.add_argument("--strictness", default="strict", choices=["strict", "normal", "loose"],
                        help="Profile from config.py (controls embedding batch size)")
    parser.add_argument("--device", default=None, help="torch device: cuda, mps, or cpu (auto if unset)")
    args = parser.parse_args()

    prof = get_profile(args.strictness)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device} | embed_batch={prof.embed_batch}")

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, y, inv_map = list_images_with_labels(data_dir)
    if len(paths) == 0:
        raise SystemExit(f"No images found under {data_dir}")

    print(f"[INFO] Found {len(paths)} images across {len(set(y))} classes.")
    imgs: List[Optional[Image.Image]] = load_images(paths)
    X = embed_images(imgs, device=device, batch_size=prof.embed_batch)

    # Guard for tiny classes in KNN
    counts = {cls: y.count(cls) for cls in set(y)}
    min_class = min(counts.values())
    k = max(1, min(5, min_class - 1)) if min_class > 1 else 1
    if k < 1:
        k = 1

    # Models to compare via CV (StratifiedKFold)
    models = {
        f"knn{k}": KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
        "lin_svm": Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", LinearSVC())])
    }
    cv = StratifiedKFold(n_splits=min(5, min_class)) if min_class >= 3 else StratifiedKFold(n_splits=2)
    scores = {}

    for name, clf in models.items():
        try:
            sc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            scores[name] = float(sc.mean())
            print(f"[CV] {name}: {sc.mean():.3f} ± {sc.std():.3f} (n={len(sc)})")
        except Exception as e:
            print(f"[WARN] CV failed for {name}: {e}")

    if not scores:
        print("[WARN] No CV scores produced; defaulting to lin_svm")
        best_name = "lin_svm"
    else:
        best_name = max(scores, key=scores.get)

    best_clf = models[best_name]
    best_clf.fit(X, y)
    print(f"[INFO] Selected: {best_name}")

    # Save artifacts
    model_path = out_dir / "face_classifier.joblib"
    joblib.dump(best_clf, model_path)
    print(f"[SAVE] {model_path}")

    # Label map
    labelmap = {inv_map[i]: i for i in inv_map}  # name->int
    with (out_dir / "labelmap.json").open("w", encoding="utf-8") as f:
        json.dump(labelmap, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_dir / 'labelmap.json'}")

    # Centroids (for distance-based analysis later)
    centroids = compute_class_centroids(X, y)
    with (out_dir / "centroids.json").open("w", encoding="utf-8") as f:
        json.dump(centroids, f)
    print(f"[SAVE] {out_dir / 'centroids.json'}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
