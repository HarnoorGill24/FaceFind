# train_face_classifier.py
# pip install facenet-pytorch torch torchvision scikit-learn joblib pillow
import json, os, random, sys, glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import dump

# --------------------
# CONFIG
# --------------------
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/people")
OUT_DIR  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_IMAGES_PER_CLASS = 2      # skip classes with fewer images
USE_EXISTING_EMB = True       # if .npy with same stem exists, use it; else compute
N_FOLDS = 5                   # CV folds

# --------------------
# DEVICE
# --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# --------------------
# EMBEDDING MODEL
# --------------------
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
tfm = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),         # [0,1]
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # to [-1,1]
])

def embed_image(img_path: Path) -> np.ndarray:
    # Try cached .npy with same stem
    if USE_EXISTING_EMB:
        npy = img_path.with_suffix(".npy")
        if npy.exists():
            return np.load(npy)

    img = Image.open(img_path).convert("RGB")
    t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = resnet(t).cpu().numpy()[0]
    return vec

# --------------------
# LOAD DATA
# --------------------
classes = [d for d in DATA_DIR.iterdir() if d.is_dir()]
classes = [c for c in classes if c.name.lower() not in ("unknown","misc","mixed")]
X, y, files = [], [], []

label2id = {}
for i, cls in enumerate(sorted(classes, key=lambda p: p.name.lower())):
    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp"):
        imgs.extend(glob.glob(str(cls / ext)))
    if len(imgs) < MIN_IMAGES_PER_CLASS:
        print(f"Skipping '{cls.name}' (only {len(imgs)} images).")
        continue
    label2id[cls.name] = len(label2id)

    for p in imgs:
        vec = embed_image(Path(p))
        X.append(vec)
        y.append(label2id[cls.name])
        files.append(p)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

if len(np.unique(y)) < 2:
    raise SystemExit("Need at least 2 labeled people with enough images to train a classifier.")

print(f"Loaded {len(X)} embeddings across {len(np.unique(y))} classes.")

# --------------------
# CROSS-VALIDATION: kNN vs Linear SVM
# --------------------
def cv_score(clf):
    skf = StratifiedKFold(n_splits=min(N_FOLDS, np.min(np.bincount(y)) if len(np.unique(y))>1 else 2), shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], pred))
    return float(np.mean(scores)), clf

candidates = [
    ("knn3", KNeighborsClassifier(n_neighbors=1, weights="distance")),
    ("knn5", KNeighborsClassifier(n_neighbors=3, weights="distance")),
    ("lin_svm", SVC(kernel="linear", probability=True, class_weight="balanced")),
]

best_name, best_score, best_clf = None, -1.0, None
for name, clf in candidates:
    try:
        score, _ = cv_score(clf)
        print(f"{name} CV acc: {score:.3f}")
        if score > best_score:
            best_name, best_score, best_clf = name, score, clf
    except Exception as e:
        print(f"{name} failed CV: {e}")

# Fit best on all data
best_clf.fit(X, y)
print(f"Selected: {best_name} (CV acc ~ {best_score:.3f})")

# --------------------
# SAVE ARTIFACTS
# --------------------
dump(best_clf, OUT_DIR / "face_classifier.joblib")
with open(OUT_DIR / "labelmap.json", "w") as f:
    json.dump({int(v):k for k,v in label2id.items()}, f, indent=2)

# Optional: save class centroids for thresholding / “unknown”
centroids = {}
for cls_id in np.unique(y):
    centroids[int(cls_id)] = X[y==cls_id].mean(axis=0).tolist()
with open(OUT_DIR / "centroids.json", "w") as f:
    json.dump(centroids, f)

print("Saved:", OUT_DIR / "face_classifier.joblib", OUT_DIR / "labelmap.json", OUT_DIR / "centroids.json")
