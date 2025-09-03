#!/usr/bin/env python3
"""Predict face labels for images and write a results CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from facefind.cli_common import (
    add_config_profile,
    add_device,
    add_log_level,
    add_version,
    validate_path,
)
from facefind.embedding_utils import embed_images, get_device, load_images
from facefind.io_schema import PREDICTIONS_SCHEMA, SCHEMA_MAGIC
from facefind.utils import IMAGE_EXTS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL import Image


def list_images(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]


def softmax_row(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a single row vector."""
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - import failure
        raise RuntimeError(
            "NumPy is required. Install with `pip install -r requirements.txt`."
        ) from e

    m = np.max(x)
    e = np.exp(x - m)
    s = e.sum()
    if s <= 0.0 or not math.isfinite(s):
        return np.full_like(x, 1.0 / len(x), dtype=np.float64)
    return e / s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="facefind-predict", description="Predict face labels for images/crops"
    )
    add_version(ap)
    ap.add_argument("--input", required=True, help="Image file or directory of images to classify")
    ap.add_argument(
        "--models-dir", required=True, help="Directory with face_classifier.joblib + labelmap.json"
    )
    ap.add_argument("--output", required=True, help="Output CSV path")
    add_device(ap)
    add_config_profile(ap)
    add_log_level(ap)
    args = ap.parse_args(argv)

    level = getattr(logging, args.log_level, logging.INFO)
    logging.basicConfig(level=level, force=True)

    try:
        import joblib
        import numpy as np
        from PIL import Image  # noqa: F401
    except Exception as e:  # pragma: no cover - import failure
        raise SystemExit(
            "NumPy, joblib, and Pillow are required. "
            "Install with `pip install -r requirements.txt`."
        ) from e

    img_root = validate_path(Path(args.input).expanduser().resolve(), kind="input")
    model_dir = validate_path(Path(args.models_dir).expanduser().resolve(), kind="models-dir")
    out_csv = Path(args.output).expanduser().resolve()

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
    logger.info("Using device: %s", device)

    # Load images → embeddings
    logger.info("Loading %d images…", len(paths))
    pil_list: list[Image.Image | None] = load_images(paths)
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
        w.writerow(
            list(PREDICTIONS_SCHEMA) + ["pred_index", "raw_score"]
        )  # extras are helpful for debugging

        for i, p in enumerate(paths):
            idx = int(indices[i])
            best_label = inv_map.get(idx, str(idx))
            best_prob = float(probs[i, idx])
            best_raw = float(raw_scores[i, idx]) if raw_scores.ndim == 2 else best_prob

            # Canonical first: path,label,prob
            w.writerow([str(p), best_label, f"{best_prob:.6f}", idx, f"{best_raw:.6f}"])

    logger.info("Wrote predictions → %s", out_csv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
