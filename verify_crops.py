#!/usr/bin/env python3
"""
FaceFind - verify_crops.py
Re-run face detection on saved crops to filter out false positives.
Writes a filtered manifest and optionally moves rejects.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

cv2: Optional[Any]
try:
    import cv2 as _cv2
    cv2 = _cv2
except Exception:
    cv2 = None

from PIL import Image
from facenet_pytorch import MTCNN

from config import get_profile
from embedding_utils import get_device


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Verify crops by re-detecting faces, reject false positives")
    parser.add_argument("crops_dir", help="Directory of face crops to verify")
    parser.add_argument("--reject-dir", help="Where to move rejects (optional)")
    parser.add_argument("--strictness", default="strict", choices=["strict", "normal", "loose"],
                        help="Profile from config.py (controls thresholds)")
    parser.add_argument("--device", default=None, help="torch device: cuda, mps, or cpu (auto if unset)")
    args = parser.parse_args()

    prof = get_profile(args.strictness)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Workaround for MPS AdaptivePool bug
    mtcnn_device = device
    if device == "mps":
        print("[INFO] Detectors on MPS can fail due to adaptive pooling. Using CPU for MTCNN.")
        mtcnn_device = "cpu"

    mtcnn = MTCNN(
        keep_all=True,
        thresholds=prof.mtcnn_thresholds,
        min_face_size=prof.min_size,
        device=mtcnn_device,
    )

    crops_dir = Path(args.crops_dir).expanduser().resolve()
    reject_dir = Path(args.reject_dir).expanduser().resolve() if args.reject_dir else None
    if reject_dir:
        ensure_dir(reject_dir)

    manifest_rows = []
    kept, rejected = 0, 0

    for img_path in sorted(crops_dir.glob("*.jpg")):
        try:
            pil = Image.open(img_path).convert("RGB")
            boxes, probs = mtcnn.detect(pil)
            if boxes is None or probs is None or len(boxes) == 0:
                # Reject
                if reject_dir:
                    shutil.move(str(img_path), reject_dir / img_path.name)
                rejected += 1
                continue
            # Keep
            manifest_rows.append([str(img_path), f"{max(probs):.4f}"])
            kept += 1
        except Exception as e:
            print(f"[WARN] verify failed on {img_path}: {e}", file=sys.stderr)
            if reject_dir:
                shutil.move(str(img_path), reject_dir / img_path.name)
            rejected += 1

    out_csv = crops_dir.parent.parent / "crops_verified.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "prob"])
        writer.writerows(manifest_rows)

    print(f"[INFO] Verification complete. Kept: {kept}, Rejected: {rejected}")
    print(f"[INFO] Filtered manifest: {out_csv}")
    if reject_dir:
        print(f"[INFO] Rejects: {reject_dir}")


if __name__ == "__main__":
    main()
