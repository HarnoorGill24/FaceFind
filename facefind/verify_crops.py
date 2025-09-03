#!/usr/bin/env python3
"""
FaceFind - verify_crops.py
Re-run face detection on saved crops to filter out false positives.
Writes a filtered manifest and optionally moves rejects.
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image

from facefind.config import get_profile
from facefind.embedding_utils import get_device
from facefind.quality import passes_quality
from facefind.utils import ensure_dir


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify crops by re-detecting faces, reject false positives")
    parser.add_argument("crops_dir", help="Directory of face crops to verify")
    parser.add_argument("--reject-dir", help="Where to move rejects (optional)")
    parser.add_argument(
        "--strictness",
        default="strict",
        choices=["strict", "normal", "loose"],
        help="Profile from config.py (controls thresholds)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device: cuda, mps, or cpu (auto if unset)",
    )
    parser.add_argument(
        "--min-var",
        type=float,
        default=100.0,
        help="Variance of Laplacian threshold to accept",
    )
    parser.add_argument(
        "--exposure-tol",
        type=float,
        default=0.05,
        help="Max fraction of under/over-exposed pixels",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prof = get_profile(args.strictness)

    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # Workaround for MPS AdaptivePool bug
    mtcnn_device = device
    if device == "mps":
        logger.info("Detectors on MPS can fail due to adaptive pooling. Using CPU for MTCNN.")
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

    kept, rejected = 0, 0

    out_csv = crops_dir.parent.parent / "crops_verified.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "prob"])
        for img_path in sorted(crops_dir.glob("*.jpg")):
            try:
                with Image.open(img_path) as pil:
                    pil = pil.convert("RGB")
                    boxes, probs = mtcnn.detect(pil)
                    if boxes is None or probs is None or len(boxes) == 0:
                        if reject_dir:
                            shutil.move(str(img_path), reject_dir / img_path.name)
                        rejected += 1
                        continue

                    ok, var, exposure = passes_quality(
                        pil, min_var=args.min_var, exposure_tol=args.exposure_tol
                    )
                    if not ok:
                        logger.debug(
                            "Rejected %s for quality: var=%.2f exposure=%s",
                            img_path,
                            var,
                            exposure,
                        )
                        if reject_dir:
                            shutil.move(str(img_path), reject_dir / img_path.name)
                        rejected += 1
                        continue
                # Keep
                writer.writerow([str(img_path), f"{max(probs):.4f}"])
                f.flush()
                kept += 1
            except Exception as e:  # pragma: no cover
                logger.warning("verify failed on %s: %s", img_path, e)
                if reject_dir:
                    shutil.move(str(img_path), reject_dir / img_path.name)
                rejected += 1

    logger.info("Verification complete. Kept: %d, Rejected: %d", kept, rejected)
    logger.info("Filtered manifest: %s", out_csv)
    if reject_dir:
        logger.info("Rejects: %s", reject_dir)


if __name__ == "__main__":
    main()
