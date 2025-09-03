#!/usr/bin/env python3
"""
FaceFind - split_clusters.py

Split a clustering or prediction CSV into per-label folders.
Creates hard links by default (use --copy to actually copy files).

CSV column detection (first match wins):
- path/file/image
- cluster/label/prediction
- confidence (optional; ignored here)
"""
import argparse
import csv
import logging
import os
import shutil
from pathlib import Path

from facefind.utils import sanitize_label
from utils.common import ensure_dir

IMAGE_COL_CANDIDATES = ("path", "file", "image")
LABEL_COL_CANDIDATES = ("cluster", "label", "prediction")


def place(dst_root: Path, safe_label: str, src: Path, copy: bool) -> None:
    dst_dir = dst_root / safe_label
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)  # hard link saves space
    except Exception:
        shutil.copy2(src, dst)


logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Split images into folders by cluster/prediction")
    ap.add_argument("csv_path", help="CSV with image path + cluster/label columns")
    ap.add_argument("out_dir", help="Destination directory for per-label folders")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking")
    ap.add_argument("--rel-root", default=None, help="Optional root to resolve relative CSV paths")
    ap.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO)")
    args = ap.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)
    csv_path = Path(args.csv_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    rel_root = Path(args.rel_root).expanduser().resolve() if args.rel_root else None

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}

        def pick(cands):
            for c in cands:
                if c in headers:
                    return headers[c]
            return None

        img_col = pick(IMAGE_COL_CANDIDATES)
        lab_col = pick(LABEL_COL_CANDIDATES)

        if not img_col or not lab_col:
            raise SystemExit(
                f"CSV must contain image column in {IMAGE_COL_CANDIDATES} " f"and label column in {LABEL_COL_CANDIDATES}. Found: {reader.fieldnames}"
            )

        rows = list(reader)

    placed = 0
    skipped = 0

    for row in rows:
        raw_path = (row.get(img_col) or "").strip()
        label = (row.get(lab_col) or "").strip()
        if not raw_path or not label:
            skipped += 1
            continue

        p = Path(raw_path)
        if not p.is_absolute() and rel_root:
            p = (rel_root / p).resolve()

        if not p.exists():
            skipped += 1
            continue

        safe_label = sanitize_label(label)
        place(out_dir, safe_label, p, copy=args.copy)
        placed += 1

    logger.info("Placed: %d, Skipped: %d", placed, skipped)
    logger.info("Out dir: %s", out_dir)


if __name__ == "__main__":
    main()
