#!/usr/bin/env python3
"""
FaceFind - split_clusters.py

Split a clustering or prediction CSV into per-label folders.
Places images into directories named after each cluster/label by creating
hard links (or copies with --copy).
"""
import argparse
import csv
import os
import re
import shutil
from pathlib import Path


def safe_name(name: str) -> str:
    """Sanitize a cluster/prediction name for filesystem use."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Split images into folders by cluster/prediction")
    ap.add_argument("csv_path", help="CSV file with columns: path and cluster/prediction")
    ap.add_argument("out_dir", help="Destination directory for per-cluster folders")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of hard linking")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def place(cluster: str, src: Path) -> None:
        dest = out_dir / safe_name(cluster)
        dest.mkdir(parents=True, exist_ok=True)
        dst = dest / src.name
        if args.copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)  # hardlink to save space
            except Exception:
                shutil.copy2(src, dst)

    placed = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = (row.get("path") or row.get("crop_path") or row.get("file") or row.get("image") or "").strip()
            cl = (
                row.get("cluster")
                or row.get("cluster_id")
                or row.get("pred_label")
                or row.get("label")
                or row.get("prediction")
                or ""
            ).strip()
            if not p or not cl:
                continue
            src = Path(p).expanduser()
            if not src.is_absolute():
                src = (csv_path.parent / src).resolve()
            if not src.exists():
                continue
            try:
                place(cl, src)
                placed += 1
            except Exception:
                pass

    print(f"[INFO] Placed {placed} files into {out_dir}")


if __name__ == "__main__":
    main()
