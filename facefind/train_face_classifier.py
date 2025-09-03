#!/usr/bin/env python3
"""
FaceFind - train_face_classifier.py
Create embeddings from a labeled folder tree and train a simple classifier.
Select the best model via cross-validation and save:
- models/face_classifier.joblib
- models/labelmap.json
- models/centroids.json  (keys as class names)
- models/embeddings.npy  (nice-to-have, for reuse/debugging)
- models/train_paths.json
- models/train_labels.json

Respects --strictness profile from config.py to set embedding batch size.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from facefind.config import get_profile
from facefind.embedding_utils import embed_images, get_device, load_images
from utils.common import IMAGE_EXTS

logger = logging.getLogger(__name__)


def list_images_with_labels(root: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
    """
    Expects structure:
      root/
        person_a/ img1.jpg, img2.jpg, ...
        person_b/ ...
    Returns: (paths, y_ints, inv_labelmap[int->name])
    """
    paths: List[Path] = []
    labels: List[int] = []
    classes = sorted([d for d in root.iterdir() if d.is_dir()])
    name_to_int: Dict[str, int] = {d.name: i for i, d in enumerate(classes)}
    inv_map: Dict[int, str] = {i: name for name, i in name_to_int.items()}

    for cls_dir in classes:
        cls_idx = name_to_int[cls_dir.name]
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
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
    parser.add_argument("--strictness", default="strict", choices=["strict", "normal", "loose"], help="Profile from config.py (controls embedding batch size)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="torch device: cuda, mps, or cpu (auto if unset)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)
    prof = get_profile(args.strictness)
    device = get_device(args.device)
    logger.info("Using device: %s | embed_batch=%s", device, prof.embed_batch)

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, y, inv_map = list_images_with_labels(data_dir)
    if len(paths) == 0:
        raise SystemExit(f"No images found under {data_dir}")

    logger.info("Found %d images across %d classes.", len(paths), len(set(y)))
    logger.info("Loading images...")
    imgs: List[Optional[Image.Image]] = load_images(paths)

    # Filter out failed image loads
    valid_pairs = [(p, im, lab) for p, im, lab in zip(paths, imgs, y) if im is not None]
    if len(valid_pairs) < len(paths):
        logger.warning("Dropped %d unreadable images.", len(paths) - len(valid_pairs))
    if not valid_pairs:
        raise SystemExit("No valid images to embed after filtering.")

    paths = [p for p, im, lab in valid_pairs]
    imgs = [im for p, im, lab in valid_pairs]
    y = [lab for p, im, lab in valid_pairs]

    logger.info("Embedding %d images...", len(paths))
    X = embed_images(imgs, device=device, batch_size=prof.embed_batch)

    # L2-normalize embeddings (idempotent if already normalized)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    X = X / norm

    # Save training artifacts for reproducibility / debugging
    np.save(out_dir / "embeddings.npy", X)
    with (out_dir / "train_paths.json").open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in paths], f, indent=2)
    with (out_dir / "train_labels.json").open("w", encoding="utf-8") as f:
        json.dump(y, f, indent=2)

    # Guard for tiny classes
    counts = {cls: y.count(cls) for cls in set(y)}
    min_class = min(counts.values())
    k = 1 if min_class <= 1 else max(1, min(5, min_class - 1))

    # Choose / evaluate models
    scores = {}
    if min_class < 2:
        logger.warning("Some class has only 1 sample; skipping CV and defaulting to lin_svm.")
        best_name = "lin_svm"
        best_clf = Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", LinearSVC(class_weight="balanced", max_iter=10000))])
    else:
        cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
        models = {
            f"knn{k}": KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1),
            "lin_svm": Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", LinearSVC(class_weight="balanced", max_iter=10000))]),
        }

        for name, clf in models.items():
            try:
                sc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                scores[name] = float(sc.mean())
                logger.info("[CV] %s: %.3f ± %.3f (n=%d)", name, sc.mean(), sc.std(), len(sc))
            except Exception as e:
                logger.warning("CV failed for %s: %s", name, e)

        best_name = max(scores, key=scores.get) if scores else "lin_svm"
        best_clf = models[best_name]

    # Fit best model on full data
    best_clf.fit(X, y)
    logger.info("Selected: %s", best_name)

    # Save model
    model_path = out_dir / "face_classifier.joblib"
    joblib.dump(best_clf, model_path)
    logger.info("Saved %s", model_path)

    # Label map (name -> int)
    labelmap = {inv_map[i]: i for i in inv_map}  # name->int
    with (out_dir / "labelmap.json").open("w", encoding="utf-8") as f:
        json.dump(labelmap, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", out_dir / "labelmap.json")

    # Centroids with class names as keys (easier to read)
    centroids = compute_class_centroids(X, y)  # int->vector
    name_by_int = {i: name for i, name in inv_map.items()}
    centroids_named = {name_by_int[int_k]: v for int_k, v in centroids.items()}
    with (out_dir / "centroids.json").open("w", encoding="utf-8") as f:
        json.dump(centroids_named, f, indent=2)
    logger.info("Saved %s", out_dir / "centroids.json")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
