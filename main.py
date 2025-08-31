
#!/usr/bin/env python3
"""
FaceFind - main.py
Scan images/videos, detect faces, save crops + manifest.
Uses strictness profiles from config.py (MTCNN thresholds, min size, embed batch size placeholder).
"""
import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None

import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Import strictness profile
from config import get_profile

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def iter_media(root: Path) -> List[Path]:
    for p in sorted(root.rglob('*')):
        if p.is_file() and (is_image(p) or is_video(p)):
            yield p

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_image_bgr(path: Path):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read images and videos. pip install opencv-python")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def frame_iterator(video_path: Path, step: int):
    """Yield (frame_index, frame_bgr) every `step` frames."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) required for video processing. pip install opencv-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}", file=sys.stderr)
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()

def bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[..., ::-1]  # BGR->RGB
    return Image.fromarray(rgb)

def crop_and_save(pil_img: Image.Image, box: Tuple[int, int, int, int], out_dir: Path, stem: str, face_id: int) -> Path:
    x1, y1, x2, y2 = map(int, box)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w < 1 or h < 1:
        raise ValueError("Invalid crop size")
    crop = pil_img.crop((x1, y1, x2, y2))
    out_path = out_dir / f"{stem}_face{face_id}.jpg"
    crop.save(out_path, quality=95)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="FaceFind: scan media, detect faces, save crops + manifest")
    parser.add_argument("--input", required=True, help="Path to media folder (images and/or videos)")
    parser.add_argument("--output", default="outputs", help="Output root (default: outputs)")
    parser.add_argument("--video-step", type=int, default=5, help="Take every Nth frame from video (default: 5)")
    parser.add_argument("--strictness", default="strict", choices=["strict", "normal", "loose"],
                        help="Threshold profile from config.py")
    parser.add_argument("--device", default=None, help="torch device, e.g., cuda, mps, or cpu (auto if not set)")
    parser.add_argument("--max-per-media", type=int, default=50, help="Max faces to save per media file (safety)")
    args = parser.parse_args()

    prof = get_profile(args.strictness)

    # Select device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[INFO] Using device: {device}")

    input_dir = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    crops_dir = out_root / "crops" / "pending"
    ensure_dir(crops_dir)
    manifests_dir = out_root
    ensure_dir(manifests_dir)

    # Initialize MTCNN with profile thresholds
    mtcnn = MTCNN(keep_all=True,
                  thresholds=prof.mtcnn_thresholds,
                  min_face_size=prof.min_size,
                  device=device)

    manifest_rows = []
    t0 = time.time()
    media_count = 0
    face_total = 0

    for media_path in iter_media(input_dir):
        media_count += 1
        rel = media_path.relative_to(input_dir)
        stem = media_path.stem
        try:
            if is_image(media_path):
                img_bgr = read_image_bgr(media_path)
                pil = bgr_to_pil_rgb(img_bgr)
                boxes, probs = mtcnn.detect(pil)
                if boxes is None or probs is None:
                    continue
                saved = 0
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob is None or prob < prof.min_prob:
                        continue
                    try:
                        out_path = crop_and_save(pil, box, crops_dir, stem, i)
                        manifest_rows.append([str(out_path), str(rel), f"{prob:.4f}"])
                        face_total += 1
                        saved += 1
                        if saved >= args.max_per_media:
                            break
                    except Exception as e:
                        print(f"[WARN] crop save failed for {media_path}: {e}", file=sys.stderr)

            elif is_video(media_path):
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) required for video processing. pip install opencv-python")
                saved_from_video = 0
                for frame_idx, frame in frame_iterator(media_path, args.video_step):
                    pil = bgr_to_pil_rgb(frame)
                    boxes, probs = mtcnn.detect(pil)
                    if boxes is None or probs is None:
                        continue
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        if prob is None or prob < prof.min_prob:
                            continue
                        try:
                            out_path = crop_and_save(pil, box, crops_dir, f"{stem}_f{frame_idx}", i)
                            manifest_rows.append([str(out_path), f"{rel}#frame={frame_idx}", f"{prob:.4f}"])
                            face_total += 1
                            saved_from_video += 1
                            if saved_from_video >= args.max_per_media:
                                break
                        except Exception as e:
                            print(f"[WARN] video crop save failed for {media_path}: {e}", file=sys.stderr)
                    if saved_from_video >= args.max_per_media:
                        break
            else:
                continue
        except Exception as e:
            print(f"[WARN] Failed on {media_path}: {e}", file=sys.stderr)

    # Write manifest CSV
    manifest_csv = out_root / "crops_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "source", "prob"])
        for row in manifest_rows:
            writer.writerow(row)

    dt = time.time() - t0
    print(f"[INFO] Done. Media processed: {media_count}, faces saved: {face_total}, time: {dt:.1f}s")
    print(f"[INFO] Crops dir: {crops_dir}")
    print(f"[INFO] Manifest: {manifest_csv}")

if __name__ == "__main__":
    main()
