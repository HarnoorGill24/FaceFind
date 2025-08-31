# main.py — streaming version (incremental save + chunked embedding)
# Interactive usage:  python main.py
# Non-interactive:    python main.py --input <path> --output ./outputs --video-step 5 --strictness strict

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

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


def iter_media(root: Path) -> Iterable[Path]:
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
# Models
# ---------------------------

def build_models(device):
    # Detector on CPU for robustness; embeddings on GPU (MPS) for speed
    detector = MTCNN(keep_all=True, device="cpu", thresholds=[0.6, 0.7, 0.7])
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return detector, embedder


def detect_faces(detector, bgr: np.ndarray, min_size: int = 40):
    boxes, probs = detector.detect(bgr)
    crops, out_boxes, kept_probs = [], [], []
    if boxes is None:
        return crops, out_boxes, kept_probs

    h, w = bgr.shape[:2]
    if probs is None:
        probs_iter = [None] * len(boxes)
    else:
        probs_iter = np.asarray(probs).tolist()

    for (x1, y1, x2, y2), p in zip(boxes, probs_iter):
        if (p is None) or (p < 0.6):
            continue
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


@torch.no_grad()
def embed_crops(embedder, device, crops_bgr: List[np.ndarray], batch_size: int = 128) -> np.ndarray:
    if not crops_bgr:
        return np.zeros((0, 512), dtype=np.float32)

    tensors = []
    # Pre-alloc mean/std tensors once on CPU; move batch to device later
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    for img in crops_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        ten = (ten - mean) / std
        tensors.append(ten)

    X = torch.stack(tensors, dim=0)
    embs = []
    for i in range(0, X.shape[0], batch_size):
        b = X[i:i + batch_size].to(device)
        e = embedder(b).cpu().numpy()
        embs.append(e)
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    embs = normalize(embs, norm="l2")
    return embs


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
# Per-file processing (streaming)
# ---------------------------

def process_image(path: Path, detector, crops_dir: Path, manifest_writer, idx_start: int) -> int:
    """Detect faces in an image, write crops to disk, append rows to manifest. Returns faces written."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return 0
    crops, boxes, probs = detect_faces(detector, bgr)
    faces_written = 0
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        out_dir = crops_dir / "pending"
        ensure_dir(out_dir)
        stem = Path(path).stem
        fname = f"{stem}_face{idx_start + faces_written}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        manifest_writer.writerow({
            "src": str(path),
            "type": "image",
            "frame": -1,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "prob": probs[i] if i < len(probs) else None,
            "crop_path": str(out_path)
        })
        faces_written += 1
    return faces_written


def process_video(path: Path, detector, step: int, crops_dir: Path, manifest_writer, idx_start: int) -> int:
    faces_written = 0
    for fno, frame in frame_sampler(path, step=step):
        crops, boxes, probs = detect_faces(detector, frame)
        for i, (crop, box) in enumerate(zip(crops, boxes)):
            out_dir = crops_dir / "pending"
            ensure_dir(out_dir)
            stem = Path(path).stem
            fname = f"{stem}_f{fno}_face{idx_start + faces_written}.jpg"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            manifest_writer.writerow({
                "src": str(path),
                "type": "video",
                "frame": int(fno),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "prob": probs[i] if i < len(probs) else None,
                "crop_path": str(out_path)
            })
            faces_written += 1
    return faces_written


# ---------------------------
# CLI with interactive prompts
# ---------------------------

def resolve_dir(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


@click.command()
@click.option("--input", "input_dir", type=str,
              prompt="Input folder path (e.g., ~/Desktop/ToCull or your iCloud folder)",
              help="Root folder of images/videos")
@click.option("--output", "output_dir", type=str, prompt="Output folder path", default="./outputs",
              show_default=True, help="Where to write crops + manifests + summary")
@click.option("--video-step", type=int, prompt="Video frame step (sample every Nth frame)", default=5,
              show_default=True)
@click.option("--batch-size", type=int, prompt="Embedding batch size", default=128, show_default=True)
@click.option("--strictness", type=click.Choice(["strict", "normal", "loose"]),
              prompt="Clustering strictness", default="strict", show_default=True)
def main(input_dir: str, output_dir: str, video_step: int, batch_size: int, strictness: str):
    # Resolve paths
    input_dir = resolve_dir(input_dir)
    output_dir = resolve_dir(output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        click.echo(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(2)
    ensure_dir(output_dir)
    crops_dir = output_dir / "crops"
    ensure_dir(crops_dir)

    # Reset old manifest if present
    manifest_path = output_dir / "crops_manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()

    # Prepare models
    t0 = time.time()
    device = get_device()
    click.echo(f"[INFO] Using device: {device}")
    detector, embedder = build_models(device)

    media = list(iter_media(input_dir))
    click.echo(f"[INFO] Found {len(media)} media files under {input_dir}")

    # Pass 1: DETECT → write crops immediately, append manifest rows (streaming, low RAM)
    total_faces = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "src", "type", "frame", "x1", "y1", "x2", "y2", "prob", "crop_path"
        ])
        writer.writeheader()

        for p in tqdm(media, desc="Scanning media"):
            try:
                if p.suffix.lower() in IMAGE_EXT:
                    total_faces += process_image(p, detector, crops_dir, writer, idx_start=total_faces)
                else:
                    total_faces += process_video(p, detector, video_step, crops_dir, writer, idx_start=total_faces)
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

    if total_faces == 0:
        click.echo("[INFO] No faces detected. Exiting.")
        sys.exit(0)

    click.echo(f"[INFO] Detected {total_faces} faces. Starting embedding in chunks…")

    # Pass 2: EMBED in chunks → write to a memory-mapped array to stay within RAM
    emb_path = output_dir / "embeddings.f32.memmap"
    # Create memmap (N x 512 float32)
    emmap = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(total_faces, 512))  # disk-backed
    offset = 0
    chunk_size = max(256, batch_size)  # chunk images to load & embed

    # Iterate manifest in chunks for embedding
    for chunk in pd.read_csv(manifest_path, chunksize=chunk_size):
        crops = []
        for cp in chunk["crop_path"].tolist():
            img = cv2.imread(cp)
            if img is None:
                # Keep alignment between rows and embeddings; if missing, use zero vector
                crops.append(np.zeros((160, 160, 3), dtype=np.uint8))
            else:
                crops.append(img)
        embs = embed_crops(embedder, device, crops, batch_size=batch_size)
        n = embs.shape[0]
        emmap[offset:offset + n] = embs
        offset += n

    del emmap  # flush memmap to disk

    # Reload as mmap (read-only) and cluster
    all_embs = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(total_faces, 512))
    labels, _ = run_hdbscan(all_embs, strictness)
    # Convert to regular ndarray for downstream operations
    labels = np.array(labels, dtype=int)

    # Pass 3: WRITE final clusters.csv (stream through manifest and pair with labels)
    clusters_csv = output_dir / "clusters.csv"
    with open(manifest_path, "r") as src_f, open(clusters_csv, "w", newline="") as out_f:
        reader = csv.DictReader(src_f)
        writer = csv.DictWriter(out_f, fieldnames=[
            "cluster", "label", "type", "src", "frame", "x1", "y1", "x2", "y2", "prob", "crop_path"
        ])
        writer.writeheader()
        i = 0
        for row in reader:
            lab = labels[i] if i < len(labels) else -1
            cluster_name = "unknown" if lab == -1 else f"cluster_{lab:04d}"
            row_out = {
                "cluster": cluster_name,
                "label": int(lab),
                "type": row["type"],
                "src": row["src"],
                "frame": int(row["frame"]),
                "x1": int(float(row["x1"])), "y1": int(float(row["y1"])),
                "x2": int(float(row["x2"])), "y2": int(float(row["y2"])),
                "prob": (None if row["prob"] == "" else float(row["prob"])),
                "crop_path": row["crop_path"]
            }
            writer.writerow(row_out)
            i += 1

    # Summary
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    summary = {
        "input": str(input_dir),
        "device": str(device),
        "files_scanned": len(media),
        "faces_detected": int(total_faces),
        "clusters": n_clusters,
        "noise": n_noise,
        "strictness": strictness,
        "video_step": video_step,
        "batch_size": batch_size,
        "runtime_sec": round(time.time() - t0, 2)
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"[DONE] Wrote crops under {crops_dir}/pending")
    click.echo(f"[DONE] Manifest  : {manifest_path}")
    click.echo(f"[DONE] Embeddings: {emb_path} (memmap)")
    click.echo(f"[DONE] Clusters  : {clusters_csv}")
    click.echo(f"[SUMMARY] {n_clusters} clusters, {n_noise} noise faces")


if __name__ == "__main__":
    # Optional: silence urllib3 LibreSSL warning (cosmetic)
    try:
        import warnings
        from urllib3.exceptions import NotOpenSSLWarning

        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass
    main()
