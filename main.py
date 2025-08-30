# main.py (interactive prompts enabled)
# Usage (interactive):  python main.py
# Usage (non-interactive): python main.py --input <path> --output ./outputs --video-step 5 --strictness strict

import os
import sys
import math
import time
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import click
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
import hdbscan

# ---------------------------
# Utility: device & I/O
# ---------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

def iter_media(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT.union(VIDEO_EXT):
            yield p

def frame_sampler(video_path: Path, step: int = 5, max_frames: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    count = 0
    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fno += 1
        if fno % step != 0:
            continue
        yield fno, frame
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Face detection & embedding
# ---------------------------

def build_models(device):
    detector = MTCNN(keep_all=True, device="cpu", thresholds=[0.6, 0.7, 0.7])
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return detector, embedder

@torch.no_grad()
def embed_crops(embedder, device, crops_bgr, batch_size: int = 128) -> np.ndarray:
    tensors = []
    for img in crops_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        ten = (ten - mean) / std
        tensors.append(ten)
    if not tensors:
        return np.zeros((0, 512), dtype=np.float32)
    X = torch.stack(tensors, dim=0).to(device)

    embs = []
    for i in range(0, X.shape[0], batch_size):
        b = X[i:i + batch_size]
        e = embedder(b).cpu().numpy()
        embs.append(e)
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    embs = normalize(embs, norm="l2")
    return embs

def detect_faces(detector, bgr: np.ndarray, min_size: int = 40):
    boxes, probs = detector.detect(bgr)
    crops, out_boxes, kept_probs = [], [], []
    if boxes is None:
        return crops, out_boxes, kept_probs

    h, w = bgr.shape[:2]
    # Ensure probs is iterable of floats (or None)
    # facenet_pytorch returns numpy arrays; convert to list safely
    if probs is None:
        probs_iter = [None] * len(boxes)
    else:
        # tolist() avoids numpy truthiness issues
        probs_iter = np.asarray(probs).tolist()

    for (x1, y1, x2, y2), p in zip(boxes, probs_iter):
        # filter by score
        if (p is None) or (p < 0.6):
            continue
        # clip and size filter
        x1, y1, x2, y2 = map(int, (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append(crop)
        out_boxes.append((x1, y1, x2, y2))
        kept_probs.append(float(p) if p is not None else None)

    return crops, out_boxes, kept_probs

# ---------------------------
# Clustering
# ---------------------------

def strictness_to_hdbscan_params(strictness: str):
    s = strictness.lower()
    if s == "strict":
        return dict(min_cluster_size=12, min_samples=8, cluster_selection_epsilon=0.0)
    if s == "normal":
        return dict(min_cluster_size=8, min_samples=5, cluster_selection_epsilon=0.05)
    if s == "loose":
        return dict(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.1)
    return dict(min_cluster_size=8, min_samples=5, cluster_selection_epsilon=0.05)

def run_hdbscan(embs: np.ndarray, strictness: str):
    if embs.shape[0] == 0:
        return np.array([], dtype=int), None
    params = strictness_to_hdbscan_params(strictness)
    clusterer = hdbscan.HDBSCAN(metric="euclidean", **params)
    labels = clusterer.fit_predict(embs)
    return labels, clusterer

# ---------------------------
# Per-file processing
# ---------------------------

def process_image(path: Path, detector):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return []
    crops, boxes, probs = detect_faces(detector, bgr)
    out = []
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        out.append({"type": "image", "src": str(path), "frame": -1, "box": box,
                    "prob": (probs[i] if i < len(probs) else None), "crop": crop})
    return out

def process_video(path: Path, detector, step: int):
    results = []
    for fno, frame in frame_sampler(path, step=step):
        crops, boxes, probs = detect_faces(detector, frame)
        for i, (crop, box) in enumerate(zip(crops, boxes)):
            results.append({"type": "video", "src": str(path), "frame": int(fno), "box": box,
                            "prob": (probs[i] if i < len(probs) else None), "crop": crop})
    return results

# ---------------------------
# CLI with interactive prompts
# ---------------------------

def resolve_dir(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

@click.command()
@click.option(
    "--input",
    "input_dir",
    type=str,
    prompt="Input folder path (e.g., ~/Desktop/ToCull or your iCloud folder)",
    help="Root folder of images/videos",
)
@click.option(
    "--output",
    "output_dir",
    type=str,
    prompt="Output folder path",
    default="./outputs",
    show_default=True,
    help="Where to write crops + clusters.csv + summary.json",
)
@click.option(
    "--video-step",
    type=int,
    prompt="Video frame step (sample every Nth frame)",
    default=5,
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    prompt="Embedding batch size",
    default=128,
    show_default=True,
)
@click.option(
    "--strictness",
    type=click.Choice(["strict", "normal", "loose"]),
    prompt="Clustering strictness",
    default="strict",
    show_default=True,
)
def main(input_dir: str, output_dir: str, video_step: int, batch_size: int, strictness: str):
    # Resolve and validate paths
    input_dir = resolve_dir(input_dir)
    output_dir = resolve_dir(output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        click.echo(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = get_device()
    click.echo(f"[INFO] Using device: {device}")
    detector, embedder = build_models(device)

    media = list(iter_media(input_dir))
    click.echo(f"[INFO] Found {len(media)} media files under {input_dir}")

    # Pass 1: detect & collect crops
    all_items = []
    for p in tqdm(media, desc="Scanning media"):
        try:
            if p.suffix.lower() in IMAGE_EXT:
                items = process_image(p, detector)
            else:
                items = process_video(p, detector, step=video_step)
            all_items.extend(items)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not all_items:
        click.echo("[INFO] No faces detected. Exiting.")
        sys.exit(0)

    # Embeddings
    crops = [it["crop"] for it in all_items]
    embs = embed_crops(embedder, device, crops, batch_size=batch_size)

    # Clustering
    labels, _ = run_hdbscan(embs, strictness)
    n_noise = int(np.sum(labels == -1)) if labels.size else 0
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0)) if labels.size else 0
    click.echo(f"[INFO] Clusters: {n_clusters}   Noise (unclustered): {n_noise}")

    # Save crops + manifest
    crops_dir = output_dir / "crops"
    ensure_dir(crops_dir)
    rows = []
    for idx, (it, lab) in enumerate(zip(all_items, labels)):
        cluster_name = "unknown" if lab == -1 else f"cluster_{lab:04d}"
        out_dir = crops_dir / cluster_name
        ensure_dir(out_dir)
        stem = Path(it["src"]).stem
        if it["type"] == "image":
            fname = f"{stem}_face{idx}.jpg"
        else:
            fname = f"{stem}_f{it['frame']}_face{idx}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), it["crop"], [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        rows.append({
            "cluster": cluster_name, "label": int(lab),
            "type": it["type"], "src": it["src"], "frame": it["frame"],
            "x1": it["box"][0], "y1": it["box"][1], "x2": it["box"][2], "y2": it["box"][3],
            "prob": it["prob"], "crop_path": str(out_path)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "clusters.csv", index=False)
    summary = {
        "input": str(input_dir), "device": str(device), "files_scanned": len(media),
        "faces_detected": len(all_items), "clusters": n_clusters, "noise": n_noise,
        "strictness": strictness, "video_step": video_step, "batch_size": batch_size,
        "runtime_sec": round(time.time() - t0, 2)
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"[DONE] Wrote crops to {crops_dir}")
    click.echo(f"[DONE] Manifest: {output_dir / 'clusters.csv'}")
    click.echo(f"[DONE] Summary  : {output_dir / 'summary.json'}")

if __name__ == "__main__":
    main()
