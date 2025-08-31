# verify_crops.py
# Usage:
#   python3 verify_crops.py outputs/crops --reject-dir outputs/rejects --min-prob 0.90 --min-size 40
from pathlib import Path
from PIL import Image
import argparse, shutil, torch
from facenet_pytorch import MTCNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("--reject-dir", default="outputs/rejects")
    ap.add_argument("--min-prob", type=float, default=0.90)
    ap.add_argument("--min-size", type=int, default=40)  # px on the shorter side
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.7, 0.8, 0.92])

    in_dir = Path(args.in_dir)
    rej_dir = Path(args.reject_dir); rej_dir.mkdir(parents=True, exist_ok=True)

    kept = rejected = 0
    for p in in_dir.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"): continue
        img = Image.open(p).convert("RGB")
        boxes, probs = mtcnn.detect(img)

        ok = False
        if boxes is not None:
            for (x1, y1, x2, y2), pr in zip(boxes, probs):
                if pr is None: continue
                w, h = x2 - x1, y2 - y1
                if pr >= args.min_prob and min(w, h) >= args.min_size:
                    ok = True; break

        if ok:
            kept += 1
        else:
            shutil.move(str(p), str(rej_dir / p.name))
            rejected += 1

    print(f"Kept {kept}, rejected {rejected}. Rejects: {rej_dir}")

if __name__ == "__main__":
    main()
