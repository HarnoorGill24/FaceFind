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
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from facefind.utils import ensure_dir, sanitize_label
from facefind.cli_common import add_log_level, add_version, validate_path

from facefind.io_schema import PATH_ALIASES, LABEL_ALIASES, PROB_ALIASES


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
    """Place *src* into *dst_root* under a directory for *label*.

    The label is sanitized to avoid unsafe filesystem characters.
    """

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

    return pick(PATH_ALIASES), pick(LABEL_ALIASES), pick(PROB_ALIASES)


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="facefind-apply", description="Apply predictions to organize images by confidence.")
    add_version(ap)
    ap.add_argument("--input", required=True, help="CSV with columns path,label,prob (header flexible)")
    ap.add_argument(
        "--people-dir",
        default=None,
        help="If set, accepted items also link/copy to this labeled people tree (people_by_cluster)",
    )
    ap.add_argument(
        "--output",
        default="outputs/autosort",
        help="Where to place accept/review results (default: outputs/autosort)",
    )
    ap.add_argument("--accept-threshold", type=float, default=0.80, help="Confidence >= this goes to ACCEPT (default: 0.80)")
    ap.add_argument(
        "--review-threshold",
        type=float,
        default=0.50,
        help="Confidence >= this and < accept-threshold goes to REVIEW (default: 0.50)",
    )
    ap.add_argument("--copy", action="store_true", help="Copy files instead of creating hard links")
    ap.add_argument("--rel-root", default=None, help="Resolve relative CSV paths against this root (optional)")
    ap.add_argument("--dry-run", action="store_true", help="Run without placing files")
    add_log_level(ap)
    args = ap.parse_args(argv)

    level = getattr(logging, args.log_level, logging.INFO)
    logging.basicConfig(level=level, force=True)
    csv_path = validate_path(Path(args.input).expanduser().resolve(), kind="input")
    out_root = Path(args.output).expanduser().resolve()
    accept_root = out_root / "accept"
    review_root = out_root / "review"
    if not args.dry_run:
        ensure_dir(accept_root)
        ensure_dir(review_root)

    people_dir = Path(args.people_dir).expanduser().resolve() if args.people_dir else None
    if people_dir and not args.dry_run:
        ensure_dir(people_dir)

    rel_root = Path(args.rel_root).expanduser().resolve() if args.rel_root else None
    if rel_root and not rel_root.exists():
        raise SystemExit(f"rel-root path does not exist: {rel_root}")

    if args.review_threshold > args.accept_threshold:
        logger.warning("review-threshold > accept-threshold; swapping.")
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
            raise SystemExit(
                f"CSV must contain columns: path({PATH_ALIASES}), label({LABEL_ALIASES}), prob({PROB_ALIASES}). "
                f"Found: {reader.fieldnames}"
            )

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

            safe_label = sanitize_label(label)

            if prob >= args.accept_threshold:
                if not args.dry_run:
                    place(src, accept_root, label, copy=args.copy)
                    if people_dir:
                        place(src, people_dir, label, copy=args.copy)
                accepted += 1
                by_label_accept[safe_label] = by_label_accept.get(safe_label, 0) + 1

            elif prob >= args.review_threshold:
                if not args.dry_run:
                    place(src, review_root, label, copy=args.copy)
                reviewed += 1
                by_label_review[safe_label] = by_label_review.get(safe_label, 0) + 1
            else:
                # Below review threshold → ignore (soft reject)
                rejected += 1

    # Report
    logger.info(
        "Accepted: %d, sent to review: %d, rejected(below review): %d, missing: %d",
        accepted,
        reviewed,
        rejected,
        missing,
    )
    logger.info("Accept dir: %s", accept_root)
    logger.info("Review dir: %s", review_root)
    if people_dir:
        logger.info("People dir updated: %s", people_dir)

    if by_label_accept:
        logger.info("[ACCEPT BY LABEL]")
        for k, v in sorted(by_label_accept.items()):
            logger.info("%s: %d", k, v)
    if by_label_review:
        logger.info("[REVIEW BY LABEL]")
        for k, v in sorted(by_label_review.items()):
            logger.info("%s: %d", k, v)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
