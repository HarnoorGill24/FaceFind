#!/usr/bin/env python3
"""Split clustering or prediction CSV into per-label folders."""
import argparse
import csv
import logging
import os
import shutil
from pathlib import Path

from facefind.cli_common import add_log_level, add_version, validate_path
from facefind.io_schema import LABEL_ALIASES, PATH_ALIASES, PROB_ALIASES
from facefind.utils import ensure_dir, sanitize_label

LABEL_CANDIDATES = ("cluster",) + LABEL_ALIASES
_ = PROB_ALIASES


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="facefind-split", description="Split images into folders by cluster/prediction"
    )
    add_version(ap)
    ap.add_argument("--input", required=True, help="CSV with image path + cluster/label columns")
    ap.add_argument("--output", required=True, help="Destination directory for per-label folders")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking")
    ap.add_argument("--rel-root", default=None, help="Optional root to resolve relative CSV paths")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without creating links/copies")
    add_log_level(ap)
    args = ap.parse_args(argv)

    level = getattr(logging, args.log_level, logging.INFO)
    logging.basicConfig(level=level, force=True)
    csv_path = validate_path(Path(args.input).expanduser().resolve(), kind="input")
    out_dir = Path(args.output).expanduser().resolve()
    if not args.dry_run:
        ensure_dir(out_dir)

    rel_root = Path(args.rel_root).expanduser().resolve() if args.rel_root else None
    if rel_root and not rel_root.exists():
        raise SystemExit(f"rel-root path does not exist: {rel_root}")

    placed = 0
    skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}

        def pick(cands):
            for c in cands:
                if c in headers:
                    return headers[c]
            return None

        img_col = pick(PATH_ALIASES)
        lab_col = pick(LABEL_CANDIDATES)

        if not img_col or not lab_col:
            raise SystemExit(
                f"CSV must contain image column in {PATH_ALIASES} "
                f"and label column in {LABEL_CANDIDATES}. Found: {reader.fieldnames}"
            )

        for row in reader:
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
            if not args.dry_run:
                place(out_dir, safe_label, p, copy=args.copy)
            placed += 1

    logger.info("Placed: %d, Skipped: %d", placed, skipped)
    logger.info("Out dir: %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
