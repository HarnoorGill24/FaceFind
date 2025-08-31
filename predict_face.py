# predict_face.py
# Usage:
#   Single image:
#     python3 predict_face.py outputs/crops/<some_face>.jpg --model-dir models
#   Entire folder (writes CSV):
#     python3 predict_face.py outputs/crops --model-dir models --out outputs/predictions.csv

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from joblib import load

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_embedder(device):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    def embed(img_path: Path):
        img = Image.open(img_path).convert("RGB")
        t = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(t).cpu().numpy()[0]
    return embed

def load_model(model_dir: Path):
    clf = load(model_dir / "face_classifier.joblib")
    labelmap = {int(k): v for k, v in json.load(open(model_dir / "labelmap.json")).items()}
    return clf, labelmap

def predict_one(img_path: Path, clf, labelmap, embed_fn, threshold: float):
    emb = embed_fn(img_path).reshape(1, -1)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]
        pred_id = int(np.argmax(probs))
        conf = float(np.max(probs))
    else:
        pred_id = int(clf.predict(emb)[0])
        conf = None
    name = labelmap[pred_id]
    if conf is not None and conf < threshold:
        return "unknown", conf
    return name, conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Image file or directory of images")
    ap.add_argument("--model-dir", default="models", help="Directory with face_classifier.joblib & labelmap.json")
    ap.add_argument("--threshold", type=float, default=0.60, help="Prob threshold to call as unknown")
    ap.add_argument("--out", default=None, help="If path is a directory, write CSV to this path")
    ap.add_argument("--exts", nargs="*", default=[".jpg",".jpeg",".png",".webp"], help="Extensions for folder mode")
    args = ap.parse_args()

    p = Path(args.path)
    model_dir = Path(args.model_dir)
    clf, labelmap = load_model(model_dir)

    device = get_device()
    embed_fn = build_embedder(device)

    if p.is_file():
        name, conf = predict_one(p, clf, labelmap, embed_fn, args.threshold)
        print(f"{p.name}: {name}" + (f" ({conf:.2f})" if conf is not None else ""))
    elif p.is_dir():
        imgs = [q for q in p.rglob("*") if q.suffix.lower() in [e.lower() for e in args.exts]]
        if not imgs:
            print("No images found in directory.")
            return
        rows = []
        for q in imgs:
            try:
                name, conf = predict_one(q, clf, labelmap, embed_fn, args.threshold)
                rows.append((str(q), name, "" if conf is None else f"{conf:.4f}"))
            except Exception as e:
                rows.append((str(q), "ERROR", str(e)))
        if args.out:
            outp = Path(args.out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w") as f:
                f.write("path,prediction,confidence\n")
                for r in rows: f.write(",".join(r) + "\n")
            print(f"Wrote {len(rows)} predictions to {outp}")
        else:
            # Print first few for preview
            for r in rows[:10]:
                print(", ".join(r))
            if len(rows) > 10:
                print(f"... ({len(rows)-10} more) — use --out to save all to CSV")
    else:
        print("Path not found:", p)

if __name__ == "__main__":
    main()
