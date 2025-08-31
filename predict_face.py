#!/usr/bin/env python3
"""
predict_face.py

Given a directory (or single image) of crops/images, load embeddings and a
trained classifier, and write a CSV of predictions.

The CSV is written with a commented schema line and canonical headers:
    # FaceFindPredictions,v1
    path,label,prob,pred_index,raw_score
"""
from __future__ import annotations

# ruff: noqa: E402  # allow sys.path guard before some imports
# --- robust import guard so we can import local packages even if CWD is elsewhere
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
ROOT = _THIS.parent  # repo root if this file lives at root
for extra in (ROOT, ROOT.parent):  # also try parent-of-root (just in case)
    s = str(extra)
    if s and s not in sys.path:
        sys.path.insert(0, s)
# -------------------------------------------------------------------------------

import argparse
import csv
import json
import math
from typing import List, Optional

import joblib
import numpy as np
from PIL import Image

from embedding_utils import embed_images, get_device, load_images

# Prefer the shared schema; fall back to built-ins if package not available
try:
    from facefind.io_schema import PREDICTIONS_SCHEMA, SCHEMA_MAGIC
except Exception:  # ModuleNotFoundError or anything else
    # Fallback so runs never break if the package isn't on sys.path yet
    print(
        "[WARN] 'facefind.io_schema' not found; using built-in schema. " "Create facefind/io_schema.py to share schema across tools.",
        file=sys.stderr,
    )
    PREDICTIONS_SCHEMA = ("path", "label", "prob")
    SCHEMA_MAGIC = "# FaceFindPredictions,v1"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def list_images(root: Path) -> List[Path]:
    if root.is_file() and is_image(root):
        return [root]
    return [p for p in sorted(root.rglob("*")) if is_image(p)]


def softmax_row(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a single row vector."""
    m = np.max(x)
    e = np.exp(x - m)
    s = e.sum()
    if s <= 0.0 or not math.isfinite(s):
        return np.full_like(x, 1.0 / len(x), dtype=np.float64)
    return e / s


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict face labels for images/crops")
    ap.add_argument("images", help="Image file or directory of images to classify")
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Directory with face_classifier.joblib + labelmap.json",
    )
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument(
        "--device",
        default=None,
        help="torch device: mps, cuda, or cpu (auto if unset)",
    )
    ap.add_argument(
        "--strictness",
        default="strict",
        choices=["strict", "normal", "loose"],
        help="(unused here; kept for CLI consistency with other tools)",
    )
    args = ap.parse_args()

    img_root = Path(args.images).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()

    clf_path = model_dir / "face_classifier.joblib"
    map_path = model_dir / "labelmap.json"
    if not clf_path.exists():
        raise SystemExit(f"Missing classifier: {clf_path}")
    if not map_path.exists():
        raise SystemExit(f"Missing label map: {map_path}")

    # name->int (label string to class index)
    labelmap = json.loads(map_path.read_text(encoding="utf-8"))
    inv_map = {int(v): k for k, v in labelmap.items()}

    paths = list_images(img_root)
    if not paths:
        raise SystemExit(f"No images found under {img_root}")

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load images → embeddings
    print(f"[INFO] Loading {len(paths)} images…")
    pil_list: List[Optional[Image.Image]] = load_images(paths)
    X = embed_images(pil_list, device=device)  # embedding_utils handles MPS fallback env

    # Load classifier
    clf = joblib.load(clf_path)

    # Produce per-image class scores
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)  # (N, C)
        raw_scores = probs
        indices = np.argmax(probs, axis=1)
    elif hasattr(clf, "decision_function"):
        raw = clf.decision_function(X)  # (N, C) or (N,) for binary
        if raw.ndim == 1:
            raw = np.stack([-raw, raw], axis=1)
        probs = np.vstack([softmax_row(r) for r in raw])  # pseudo-prob via softmax
        raw_scores = raw
        indices = np.argmax(probs, axis=1)
    else:
        # Fallback: only labels, no confidences
        pred = clf.predict(X)
        indices = np.asarray(pred, dtype=int)
        C = max(len(inv_map), 1)
        probs = np.full((len(paths), C), 1.0 / C, dtype=np.float64)
        raw_scores = probs

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Schema comment then canonical header
        w.writerow([SCHEMA_MAGIC])
        w.writerow(list(PREDICTIONS_SCHEMA) + ["pred_index", "raw_score"])  # extras are helpful for debugging

        for i, p in enumerate(paths):
            idx = int(indices[i])
            best_label = inv_map.get(idx, str(idx))
            best_prob = float(probs[i, idx])
            best_raw = float(raw_scores[i, idx]) if raw_scores.ndim == 2 else best_prob

            # Canonical first: path,label,prob
            w.writerow([str(p), best_label, f"{best_prob:.6f}", idx, f"{best_raw:.6f}"])

    print(f"[INFO] Wrote predictions → {out_csv}")


if __name__ == "__main__":
    main()
