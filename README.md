# FaceFind

End-to-end local pipeline for **face discovery and labeling** across large photo/video libraries.

**Pipeline:** detect → verify → cluster → (optionally split to folders) → train classifier → predict → sort predictions.

---

## Quick Start

### 1) Environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Run the full loop

**Scan (images + videos):**
```bash
python main.py \
  --input /PATH/TO/MEDIA \
  --output ./outputs \
  --video-step 5 \
  --strictness strict
```

**Verify crops (reject low-quality / non-faces):**
```bash
python verify_crops.py \
  outputs/crops \
  --reject-dir outputs/rejects \
  --strictness strict
```

**Cluster and split to per-cluster folders (optional):**
```bash
python split_clusters.py \
  outputs \
  outputs/people_by_cluster \
  --copy
```

**Train a classifier on labeled folders:**
```bash
python train_face_classifier.py \
  --data outputs/people_by_cluster \
  --out models
```

**Predict on new crops / folders:**
```bash
python predict_face.py \
  outputs/crops \
  --model-dir models \
  --out outputs/predictions.csv
```

**Apply predictions to accept/review folders (optional):**
```bash
python apply_predictions.py \
  --in outputs/predictions.csv \
  --people-dir outputs/people \
  --out-dir outputs/sorted
```

---

## Strictness Profiles (centralized in `config.py`)

Use a single flag everywhere: `--strictness {strict,normal,loose}`.

| Profile | MTCNN Thresholds (pnet / rnet / onet) | Min Face Size | Min Prob | Embed Batch |
|---|---|---|---|---|
| strict | 0.80 / 0.90 / 0.95 | 60 px | 0.95 | 64 |
| normal | 0.70 / 0.80 / 0.92 | 40 px | 0.90 | 96 |
| loose  | 0.60 / 0.70 / 0.90 | 32 px | 0.85 | 128 |

> These defaults balance recall vs precision and memory usage. Override per-script with explicit flags if you need to.

---

## Troubleshooting

- **False positives (e.g., crops of shirts):** use `--strictness strict` or raise `--min-prob 0.95` and `--min-size 60`.
- **Killed with exit code 137 (OOM):** reduce dataset / run in shards, lower `--embed-batch` (e.g., 32), or try `strict` profile.
- **kNN CV error with tiny classes:** ensure ≥3 samples per class or favor SVM fallback.
- **Slow detection:** switch hardware backend to GPU/MPS if available; consider RetinaFace/InsightFace in roadmap.

---

## Development

### Tests
Tiny tests ensure config integrity and basic environment sanity.

```bash
pip install pytest
pytest
```

### Repo Layout
- `*.py` scripts: CLI entry points for each step.
- `config.py`: strictness profiles (single source of truth).
- `models/`: trained classifier artifacts.
- `outputs/`: crops, manifests, clusters, predictions, etc.
- `tests/`: small `pytest` suite.

---

## License
MIT © 2025 <Your Name or Organization>
See [`LICENSE`](LICENSE).
