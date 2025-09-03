#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Iterator, Tuple

from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

from facefind.config import get_profile
from facefind.embedding_utils import get_device
from facefind.file_exts import VIDEO_EXTS
from facefind.utils import is_image
from utils.common import ensure_dir

# Optional dependency: OpenCV; used only for video paths.
try:
    import cv2  # type: ignore[import-not-found]
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def iter_media(root: Path) -> Iterator[Path]:
    """Yield image/video files under root in deterministic order."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and (is_image(p) or is_video(p)):
            yield p

def read_image_pil_rgb(path: Path) -> Image.Image:
    """Read still image via PIL and auto-fix EXIF orientation."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return ImageOps.exif_transpose(img)


def create_mtcnn(profile, device: str) -> MTCNN:
    """Factory to construct an MTCNN detector honoring profile and device quirks.

    On Apple Silicon, MPS can crash inside adaptive pooling; for stability we
    force the detector to run on CPU while allowing embeddings to use MPS.
    """
    mtcnn_device = device
    if device == "mps":
        logger.info("Detectors on MPS can fail due to adaptive pooling. Using CPU for MTCNN.")
        mtcnn_device = "cpu"

    return MTCNN(
        keep_all=True,
        thresholds=profile.mtcnn_thresholds,
        min_face_size=profile.min_size,
        device=mtcnn_device,
    )


def frame_iterator(video_path: Path, step: int):
    """Yield (frame_index, frame_bgr) every `step` frames."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) required for video processing. pip install opencv-python")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
            return
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def bgr_to_pil_rgb(bgr: "np.ndarray") -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV + NumPy required for BGR->RGB conversion.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def crop_and_save(
    pil_img: Image.Image,
    box: Tuple[int, int, int, int],
    out_dir: Path,
    stem: str,
    face_id: int,
) -> Path:
    x1, y1, x2, y2 = map(int, box)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w < 1 or h < 1:
        raise ValueError("Invalid crop size")
    crop = pil_img.crop((x1, y1, x2, y2))
    out_path = out_dir / f"{stem}_face{face_id}.jpg"
    crop.save(out_path, quality=95)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceFind: scan media, detect faces, save crops + manifest")
    parser.add_argument("--input", required=True, help="Path to media folder (images and/or videos)")
    parser.add_argument("--output", default="outputs", help="Output root (default: outputs)")
    parser.add_argument(
        "--video-step",
        type=int,
        default=5,
        help="Take every Nth frame from video (default: 5)",
    )
    parser.add_argument(
        "--strictness",
        default="strict",
        choices=["strict", "normal", "loose"],
        help="Threshold profile from config.py",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="torch device, e.g., cuda, mps, or cpu (auto if not set)",
    )
    parser.add_argument(
        "--max-per-media",
        type=int,
        default=50,
        help="Max faces to save per media file (safety)",
    )
    parser.add_argument(
        "--log-no-face",
        action="store_true",
        help="Log files/frames where no faces were detected",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N media files (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO)",
    )
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)

    prof = get_profile(args.strictness)

    # Shared device resolver
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    input_dir = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    crops_dir = out_root / "crops" / "pending"
    ensure_dir(crops_dir)
    manifests_dir = out_root
    ensure_dir(manifests_dir)

    # Initialize MTCNN with profile thresholds
    mtcnn = create_mtcnn(prof, device)

    t0 = time.time()
    media_count = 0
    face_total = 0
    manifest_csv = out_root / "crops_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "source", "prob"])

        # Process with graceful Ctrl-C handling
        try:
            for media_path in iter_media(input_dir):
                media_count += 1
                if args.progress_every and (media_count % args.progress_every == 0):
                    elapsed = time.time() - t0
                    logger.info("Progress: %d media processed in %.1fs", media_count, elapsed)

                rel = media_path.relative_to(input_dir)
                stem = media_path.stem
                try:
                    if is_image(media_path):
                        # Use EXIF-aware PIL path for images
                        pil = read_image_pil_rgb(media_path)
                        boxes, probs = mtcnn.detect(pil)
                        if boxes is None or probs is None or len(boxes) == 0:
                            if args.log_no_face:
                                logger.info("No faces: %s", rel)
                            continue
                        saved = 0
                        for i, (box, prob) in enumerate(zip(boxes, probs)):
                            if prob is None or prob < prof.min_prob:
                                continue
                            try:
                                out_path = crop_and_save(pil, box, crops_dir, stem, i)
                                writer.writerow([str(out_path), str(rel), f"{prob:.4f}"])
                                f.flush()
                                face_total += 1
                                saved += 1
                                if saved >= args.max_per_media:
                                    break
                            except Exception as e:  # pragma: no cover
                                logger.warning("crop save failed for %s: %s", media_path, e)

                    elif is_video(media_path):
                        if cv2 is None:
                            raise RuntimeError("OpenCV (cv2) required for video processing. pip install opencv-python")
                        saved_from_video = 0
                        for frame_idx, frame in frame_iterator(media_path, args.video_step):
                            pil = bgr_to_pil_rgb(frame)
                            boxes, probs = mtcnn.detect(pil)
                            if boxes is None or probs is None or len(boxes) == 0:
                                if args.log_no_face:
                                    logger.info("No faces: %s#frame=%d", rel, frame_idx)
                                continue
                            for i, (box, prob) in enumerate(zip(boxes, probs)):
                                if prob is None or prob < prof.min_prob:
                                    continue
                                try:
                                    out_path = crop_and_save(pil, box, crops_dir, f"{stem}_f{frame_idx}", i)
                                    writer.writerow([str(out_path), f"{rel}#frame={frame_idx}", f"{prob:.4f}"])
                                    f.flush()
                                    face_total += 1
                                    saved_from_video += 1
                                    if saved_from_video >= args.max_per_media:
                                        break
                                except Exception as e:  # pragma: no cover
                                    logger.warning("video crop save failed for %s: %s", media_path, e)
                            if saved_from_video >= args.max_per_media:
                                break
                    else:
                        continue
                except Exception as e:  # pragma: no cover
                    logger.warning("Failed on %s: %s", media_path, e)

        except KeyboardInterrupt:  # pragma: no cover
            logger.info("Interrupted. Writing partial manifest...")

    dt = time.time() - t0
    logger.info(
        "Done. Media processed: %d, faces saved: %d, time: %.1fs",
        media_count,
        face_total,
        dt,
    )
    logger.info("Crops dir: %s", crops_dir)
    logger.info("Manifest: %s", manifest_csv)


if __name__ == "__main__":
    main()
