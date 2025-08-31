#!/usr/bin/env python3
"""
FaceFind - apply_predictions.py

Read a predictions CSV (path,label,prob) and:
- Place high-confidence images into OUT/accept/<label>/...
- Place mid-confidence images into OUT/review/<label>/...
- (Optional) Update people_dir/<label>/... for accepted items (hard link by default)

CSV header is flexible:
- path/file/image
- label/prediction
- prob/score/confidence
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

IMAGE_COLS = ("path", "file", "image")
LABEL_COLS = ("label", "prediction")
PROB_COLS = ("prob", "score", "confidence")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str) -> str:
    label = (label or "").strip()
    if not label:
        return "unknown"
    # Avoid path traversal / separators
    return label.replace(os.sep, "_")


def unique_dst(dst: Path) -> Path:
    """Avoid collisions by adding -1, -2, ... if needed."""
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}-{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def place(src: Path, dst_root: Path, label: str, copy: bool) -> Path:
    safe_label = sanitize_label(label)
    dst_dir = dst_root / safe_label
    ensure_dir(dst_dir)
    dst = unique_dst(dst_dir / src.name)
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)  # hard link = fast and space-efficient
    except Exception:
        # Cross-device or FS not supporting links → fallback to copy
        shutil.copy2(src, dst)
    return dst


def detect_headers(headers) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    low = {h.lower(): h for h in headers or []}

    def pick(cands):
        for c in cands:
            if c in low:
                return low[c]
        return None

    return pick(IMAGE_COLS), pick(LABEL_COLS), pick(PROB_COLS)


def main():
    ap = argparse.ArgumentParser(description="Apply predictions to organize images by confidence.")
    ap.add_argument("csv_path", help="CSV with columns path,label,prob (header flexible)")
    ap.add_argument("--people-dir", default=None, help="If set, accepted items also link/copy to this labeled people tree (people_by_cluster)")
    ap.add_argument("--out-dir", default="outputs/autosort", help="Where to place accept/review results (default: outputs/autosort)")
    ap.add_argument("--accept-threshold", type=float, default=0.80, help="Confidence >= this goes to ACCEPT (default: 0.80)")
    ap.add_argument("--review-threshold", type=float, default=0.50, help="Confidence >= this and < accept-threshold goes to REVIEW (default: 0.50)")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of creating hard links")
    ap.add_argument("--rel-root", default=None, help="Resolve relative CSV paths against this root (optional)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    accept_root = out_root / "accept"
    review_root = out_root / "review"
    ensure_dir(accept_root)
    ensure_dir(review_root)

    people_dir = Path(args.people_dir).expanduser().resolve() if args.people_dir else None
    if people_dir:
        ensure_dir(people_dir)

    rel_root = Path(args.rel_root).expanduser().resolve() if args.rel_root else None

    if args.review_threshold > args.accept_threshold:
        print("[WARN] review-threshold > accept-threshold; swapping.", file=sys.stderr)
        args.review_threshold, args.accept_threshold = args.accept_threshold, args.review_threshold

    # Stats
    accepted = 0
    reviewed = 0
    rejected = 0
    missing = 0
    by_label_accept: Dict[str, int] = {}
    by_label_review: Dict[str, int] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        img_col, lab_col, prob_col = detect_headers(reader.fieldnames)
        if not img_col or not lab_col or not prob_col:
            raise SystemExit(f"CSV must contain columns: path({IMAGE_COLS}), label({LABEL_COLS}), prob({PROB_COLS}). " f"Found: {reader.fieldnames}")

        for row in reader:
            raw_path = (row.get(img_col) or "").strip()
            label = (row.get(lab_col) or "").strip()
            prob_s = (row.get(prob_col) or "").strip()

            if not raw_path or not label or not prob_s:
                rejected += 1
                continue

            try:
                prob = float(prob_s)
            except Exception:
                rejected += 1
                continue

            src = Path(raw_path)
            if not src.is_absolute() and rel_root:
                src = (rel_root / src).resolve()

            if not src.exists():
                missing += 1
                continue

            if prob >= args.accept_threshold:
                # ACCEPT
                place(src, accept_root, label, copy=args.copy)
                if people_dir:
                    place(src, people_dir, label, copy=args.copy)
                accepted += 1
                by_label_accept[sanitize_label(label)] = by_label_accept.get(sanitize_label(label), 0) + 1

            elif prob >= args.review_threshold:
                # REVIEW
                place(src, review_root, label, copy=args.copy)
                reviewed += 1
                by_label_review[sanitize_label(label)] = by_label_review.get(sanitize_label(label), 0) + 1
            else:
                # Below review threshold → ignore (soft reject)
                rejected += 1

    # Report
    print(f"Accepted: {accepted}, sent to review: {reviewed}, rejected(below review): {rejected}, missing: {missing}")
    print(f"Accept dir: {accept_root}")
    print(f"Review dir: {review_root}")
    if people_dir:
        print(f"People dir updated: {people_dir}")

    if by_label_accept:
        print("\n[ACCEPT BY LABEL]")
        for k, v in sorted(by_label_accept.items()):
            print(f"  {k}: {v}")
    if by_label_review:
        print("\n[REVIEW BY LABEL]")
        for k, v in sorted(by_label_review.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
