# apply_predictions.py
# Usage:
#   python3 apply_predictions.py outputs/predictions.csv \
#       --people-dir outputs/people_by_cluster \
#       --out-dir outputs/autosort \
#       --accept-threshold 0.80 \
#       --copy   # use --copy if hardlinks aren't allowed

import csv, os, shutil, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("csv_path")
ap.add_argument("--people-dir", default="outputs/people_by_cluster")
ap.add_argument("--out-dir", default="outputs/autosort")
ap.add_argument("--accept-threshold", type=float, default=0.80)
ap.add_argument("--review-low", type=float, default=0.60)
ap.add_argument("--copy", action="store_true")
args = ap.parse_args()

csv_path = Path(args.csv_path)
people_dir = Path(args.people_dir)
out_dir = Path(args.out_dir)
acc_dir = out_dir / "accept"
rev_dir = out_dir / "review"
acc_dir.mkdir(parents=True, exist_ok=True)
rev_dir.mkdir(parents=True, exist_ok=True)
people_dir.mkdir(parents=True, exist_ok=True)

def place(dst_dir, person, src):
    d = dst_dir / person
    d.mkdir(parents=True, exist_ok=True)
    dst = d / Path(src).name
    try:
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)  # hardlink saves space
    except Exception:
        shutil.copy2(src, dst)

n_accept = n_review = 0
with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        p = row.get("path") or row.get("file") or row.get("image")
        pred = (row.get("prediction") or row.get("label") or "").strip()
        conf_s = (row.get("confidence") or "").strip()
        conf = float(conf_s) if conf_s else 0.0
        if not p or not pred:
            continue
        if pred.lower() == "unknown":
            continue
        if conf >= args.accept_threshold:
            # put into accept AND also stage into people_dir so retraining sees it
            place(acc_dir, pred, p)
            place(people_dir, pred, p)
            n_accept += 1
        elif conf >= args.review_low:
            place(rev_dir, pred, p)
            n_review += 1

print(f"Accepted: {n_accept}, sent to review: {n_review}")
print(f"Accept dir: {acc_dir}")
print(f"Review dir: {rev_dir}")
print(f"People dir updated: {people_dir}")
