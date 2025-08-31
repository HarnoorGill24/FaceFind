
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
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
from facenet_pytorch import InceptionResnetV1
import joblib

from config import get_profile

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def load_images(paths: List[Path]) -> List[Image.Image]:
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception:
            out.append(None)
    return out

def embed_images(imgs: List[Image.Image], device: str, batch_size: int) -> np.ndarray:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    embs = np.zeros((len(imgs), 512), dtype=np.float32)

    def preprocess(pil):
        return pil.resize((160,160))

    idx = 0
    for chunk in batched(imgs, batch_size):
        good_idx = []
        tensors = []
        for j, im in enumerate(chunk):
            if im is None:
                continue
            try:
                im2 = preprocess(im)
                t = torch.from_numpy(np.asarray(im2)).permute(2,0,1).float() / 255.0
                tensors.append(t.unsqueeze(0))
                good_idx.append(j)
            except Exception:
                pass
        if not tensors:
            idx += len(chunk)
            continue
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = resnet(batch).cpu().numpy().astype(np.float32)
        for j_local, vec in zip(good_idx, feats):
            embs[idx + j_local, :] = vec
        idx += len(chunk)
    return embs

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

    # Device selection
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
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

    imgs = load_images(paths)
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
