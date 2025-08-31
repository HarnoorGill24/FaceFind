#!/usr/bin/env python3
"""
FaceFind - report.py
Generate a simple JSON + console summary of your dataset and pipeline status.

It looks for the following (if present):
- outputs/crops_manifest.csv          -> total crops detected
- outputs/crops/pending               -> current crops awaiting verify
- outputs/crops/rejects               -> rejected crops count
- outputs/crops_verified.csv          -> kept crops after verification
- outputs/predictions.csv             -> per-label counts and confidence stats
- models/labelmap.json                -> known classes

Usage:
  python report.py --outputs outputs --models models --predictions outputs/predictions.csv --save-json outputs/report.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def count_images_in_dir(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.reader(f))


def main():
    parser = argparse.ArgumentParser(description="Summarize FaceFind outputs and write a report JSON")
    parser.add_argument("--outputs", default="outputs", help="Outputs root (default: outputs)")
    parser.add_argument("--models", default="models", help="Models directory (default: models)")
    parser.add_argument("--predictions", default=None, help="Path to predictions CSV (optional)")
    parser.add_argument("--save-json", default=None, help="Where to save the JSON report (default: outputs/report.json)")
    args = parser.parse_args()

    outputs = Path(args.outputs).expanduser().resolve()
    models = Path(args.models).expanduser().resolve()
    preds_csv = Path(args.predictions).expanduser().resolve() if args.predictions else (outputs / "predictions.csv")

    crops_manifest = outputs / "crops_manifest.csv"
    crops_verified = outputs / "crops_verified.csv"
    crops_pending_dir = outputs / "crops" / "pending"
    crops_rejects_dir = outputs / "crops" / "rejects"
    labelmap_path = models / "labelmap.json"

    report = {}

    # Crops manifest
    rows = read_csv_rows(crops_manifest)
    total_crops = max(0, len(rows) - 1) if rows else 0
    report["crops_total_manifest"] = total_crops

    # Pending and rejects
    report["crops_pending_count"] = count_images_in_dir(crops_pending_dir)
    report["crops_rejects_count"] = count_images_in_dir(crops_rejects_dir)

    # Verified crops
    vrows = read_csv_rows(crops_verified)
    verified_count = max(0, len(vrows) - 1) if vrows else 0
    report["crops_verified_count"] = verified_count

    # Labelmap
    if labelmap_path.exists():
        try:
            with labelmap_path.open("r", encoding="utf-8") as f:
                name_to_int = json.load(f)
            report["known_classes"] = sorted(list(name_to_int.keys()))
            report["known_classes_count"] = len(name_to_int)
        except Exception:
            report["known_classes"] = []
            report["known_classes_count"] = 0
    else:
        report["known_classes"] = []
        report["known_classes_count"] = 0

    # Predictions
    pred_summary = {"total_rows": 0, "by_label": {}, "confidence_mean": None, "confidence_by_label_mean": {}}
    if preds_csv.exists():
        prows = read_csv_rows(preds_csv)
        if prows and len(prows) > 1:
            header = [h.lower() for h in prows[0]]
            idx_label = header.index("pred_label") if "pred_label" in header else None
            idx_conf = header.index("confidence") if "confidence" in header else None

            counts = Counter()
            confs = []
            confs_by_label = defaultdict(list)

            for r in prows[1:]:
                if idx_label is not None and len(r) > idx_label:
                    lbl = r[idx_label]
                    counts[lbl] += 1
                if idx_conf is not None and len(r) > idx_conf and r[idx_conf] not in ("", None):
                    try:
                        c = float(r[idx_conf])
                        confs.append(c)
                        if idx_label is not None and len(r) > idx_label:
                            confs_by_label[r[idx_label]].append(c)
                    except Exception:
                        pass

            pred_summary["total_rows"] = len(prows) - 1
            pred_summary["by_label"] = dict(counts)
            pred_summary["confidence_mean"] = round(mean(confs), 4) if confs else None
            pred_summary["confidence_by_label_mean"] = {k: round(mean(v), 4) for k, v in confs_by_label.items() if v}
    report["predictions"] = pred_summary

    # Derived metrics
    if report["crops_total_manifest"]:
        kept = report["crops_verified_count"] or 0
        report["reject_rate_pct"] = round(100.0 * (1.0 - kept / max(1, report["crops_total_manifest"])), 2)
    else:
        report["reject_rate_pct"] = None

    # Save JSON
    save_path = Path(args.save_json).expanduser().resolve() if args.save_json else (outputs / "report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[INFO] Wrote report → {save_path}")


if __name__ == "__main__":
    main()
