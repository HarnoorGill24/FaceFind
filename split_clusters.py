# split_clusters.py
# Usage:
#   python3 split_clusters.py outputs outputs/people_by_cluster --copy
# If you omit --copy, it will try hardlinks (fast, saves space) and fall back to copy if needed.

import csv, sys, os, shutil
from pathlib import Path

src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
dst_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src_dir / "people_by_cluster"
do_copy = ("--copy" in sys.argv)
crops_dir = src_dir / "crops"
clusters_csv = src_dir / "clusters.csv"
manifest_csv = src_dir / "crops_manifest.csv"

assert clusters_csv.exists(), f"Missing {clusters_csv}"
dst_dir.mkdir(parents=True, exist_ok=True)

def pick_col(header, keywords):
    header_l = [h.lower() for h in header]
    for k in keywords:
        for i,h in enumerate(header_l):
            if k in h:
                return header[i]
    return None

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

rows = read_csv(clusters_csv)
hdr = list(rows[0].keys())

cluster_col = pick_col(hdr, ["cluster", "label"])
path_col    = pick_col(hdr, ["path", "file", "crop"])
id_col      = pick_col(hdr, ["id", "crop_id", "index"])

# Optional join with manifest to resolve paths
manifest = {}
if manifest_csv.exists():
    mrows = read_csv(manifest_csv)
    mhdr = list(mrows[0].keys())
    mid_col  = pick_col(mhdr, ["id", "crop_id", "index"])
    mpath_col= pick_col(mhdr, ["path", "file", "crop"])
    if mid_col and mpath_col:
        for r in mrows:
            manifest[str(r[mid_col])] = r[mpath_col]

clusters = {}  # cluster_id -> list of absolute src paths
for r in rows:
    cid = str(r[cluster_col]) if cluster_col else None
    if not cid:
        raise SystemExit("Could not find a 'cluster' column in clusters.csv")

    # Try to resolve path
    p = None
    if path_col and r.get(path_col):
        p = r[path_col]
    elif id_col and str(r.get(id_col)) in manifest:
        p = manifest[str(r[id_col])]
    else:
        raise SystemExit("Could not resolve crop path from clusters.csv; "
                         "ensure it has a path column or provide crops_manifest.csv with ids->paths.")

    # Make absolute
    P = Path(p)
    if not P.is_absolute():
        P = (src_dir / P) if (src_dir / P).exists() else (crops_dir / P.name)
    if not P.exists():
        # last resort: try within crops dir by name
        q = crops_dir / Path(p).name
        if q.exists():
            P = q
        else:
            print(f"WARNING: missing crop file: {p}")
            continue

    clusters.setdefault(cid, []).append(P)

# Write out (hardlink or copy)
for cid, paths in clusters.items():
    out_dir = dst_dir / f"cluster_{int(cid):04d}" if cid.isdigit() else dst_dir / f"{cid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = out_dir / src.name
        if dst.exists():
            continue
        try:
            if do_copy:
                shutil.copy2(src, dst)
            else:
                os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

print(f"Done. Wrote clusters to: {dst_dir}")
