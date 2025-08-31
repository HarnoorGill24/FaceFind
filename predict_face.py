
#!/usr/bin/env python3
"""
FaceFind - predict_face.py
Load a trained classifier + labelmap, embed input images, and output predictions to CSV.
Respects --strictness profile (embedding batch size).
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import joblib

from config import get_profile
from embedding_utils import embed_images, get_device, load_images
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def main():
    parser = argparse.ArgumentParser(description="Predict faces for a folder of images/crops")
    parser.add_argument("input", help="Folder of images (e.g., outputs/crops or any folder of faces)")
    parser.add_argument("--model-dir", default="models", help="Directory with face_classifier.joblib and labelmap.json")
    parser.add_argument("--out", default="outputs/predictions.csv", help="Output CSV path")
    parser.add_argument("--strictness", default="strict", choices=["strict","normal","loose"],
                        help="Profile from config.py (controls embedding batch size)")
    parser.add_argument("--device", default=None, help="torch device: cuda, mps, or cpu (auto if unset)")
    args = parser.parse_args()

    prof = get_profile(args.strictness)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device} | embed_batch={prof.embed_batch}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_path = model_dir / "face_classifier.joblib"
    labelmap_path = model_dir / "labelmap.json"

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not labelmap_path.exists():
        raise SystemExit(f"Missing label map: {labelmap_path}")

    clf = joblib.load(model_path)
    with labelmap_path.open("r", encoding="utf-8") as f:
        name_to_int: Dict[str, int] = json.load(f)
    int_to_name = {v: k for k, v in name_to_int.items()}

    in_dir = Path(args.input).expanduser().resolve()
    paths = list_images(in_dir)
    if not paths:
        raise SystemExit(f"No images found under {in_dir}")

    imgs: List[Optional[Image.Image]] = load_images(paths)
    X = embed_images(imgs, device=device, batch_size=prof.embed_batch)

    # Predict
    # Try to obtain a confidence score:
    # - KNN: use inverse distance to nearest neighbor as a crude confidence
    # - LinearSVC: use decision_function magnitude (per class); map to pseudo-prob via softmax over distances/scores
    out_rows = [["path", "pred_label", "pred_index", "confidence"]]

    # Detect model type
    is_knn = hasattr(clf, "kneighbors")
    has_decision = hasattr(clf, "decision_function")

    if is_knn:
        # Use KNN distances for a rough confidence
        distances, indices = clf.kneighbors(X, n_neighbors=1, return_distance=True)
        preds = clf.predict(X)
        # Convert distance to pseudo-confidence
        conf = 1.0 / (1.0 + distances.reshape(-1))
        for p, yhat, c in zip(paths, preds, conf):
            out_rows.append([str(p), int_to_name.get(int(yhat), str(yhat)), int(yhat), float(c)])
    elif has_decision:
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            # binary case: convert to 2-class scores
            scores = np.vstack([-scores, scores]).T
        # softmax-like
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        for p, yhat, c in zip(paths, preds, conf):
            out_rows.append([str(p), int_to_name.get(int(yhat), str(yhat)), int(yhat), float(c)])
    else:
        preds = clf.predict(X)
        for p, yhat in zip(paths, preds):
            out_rows.append([str(p), int_to_name.get(int(yhat), str(yhat)), int(yhat), ""])

    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    print(f"[INFO] Wrote predictions → {out_csv}")

if __name__ == "__main__":
    main()
