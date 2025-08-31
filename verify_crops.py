
#!/usr/bin/env python3
"""
FaceFind - verify_crops.py
Re-check crops to reject low-probability or too-small faces using the same strictness profile.
Moves rejects to a --reject-dir and rewrites a filtered manifest.
"""
import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from config import get_profile
from embedding_utils import get_device

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}])

def main():
    parser = argparse.ArgumentParser(description="Verify crops; reject non-faces/low-prob/too-small")
    parser.add_argument("crops_dir", help="Directory containing face crops (e.g., outputs/crops/pending)")
    parser.add_argument("--reject-dir", default=None, help="Where to place rejects (default: <crops_dir>/../rejects)")
    parser.add_argument("--strictness", default="strict", choices=["strict","normal","loose"], help="Profile from config.py")
    parser.add_argument("--min-prob", type=float, default=None, help="Override min_prob (else from profile)")
    parser.add_argument("--min-size", type=int, default=None, help="Override min_size (else from profile)")
    parser.add_argument("--device", default=None, help="torch device (auto if unset)")
    args = parser.parse_args()

    prof = get_profile(args.strictness)
    min_prob = prof.min_prob if args.min_prob is None else float(args.min_prob)
    min_size = prof.min_size if args.min_size is None else int(args.min_size)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    crops_dir = Path(args.crops_dir).expanduser().resolve()
    reject_dir = Path(args.reject_dir).expanduser().resolve() if args.reject_dir else (crops_dir.parent / "rejects")
    ensure_dir(reject_dir)

    mtcnn = MTCNN(keep_all=True,
                  thresholds=prof.mtcnn_thresholds,
                  min_face_size=prof.min_size,
                  device=device)

    kept = 0
    rejected = 0
    kept_rows = []

    imgs = list_images(crops_dir)
    for p in imgs:
        try:
            pil = Image.open(p).convert("RGB")
            w, h = pil.size
            boxes, probs = mtcnn.detect(pil)
            keep = False
            if boxes is not None and probs is not None and len(probs) > 0:
                # Keep if any detection meets thresholds and is reasonably large
                for box, prob in zip(boxes, probs):
                    if prob is None or prob < min_prob:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    bw, bh = x2 - x1, y2 - y1
                    if bw >= min_size and bh >= min_size:
                        keep = True
                        break
            if keep:
                kept += 1
                kept_rows.append([str(p), f"{probs.max():.4f}" if probs is not None else ""])
            else:
                rejected += 1
                p.rename(reject_dir / p.name)
        except Exception as e:
            # On error, reject conservatively
            rejected += 1
            try:
                p.rename(reject_dir / p.name)
            except Exception:
                pass

    # Write a filtered manifest next to crops dir
    filtered_manifest = crops_dir.parent.parent / "crops_verified.csv"
    with filtered_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_path", "prob"])
        w.writerows(kept_rows)

    print(f"[INFO] Verification complete. Kept: {kept}, Rejected: {rejected}")
    print(f"[INFO] Filtered manifest: {filtered_manifest}")
    print(f"[INFO] Rejects: {reject_dir}")

if __name__ == "__main__":
    main()
