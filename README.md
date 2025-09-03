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
pip install -e .
```

### 2) Run the full loop

All CLI commands accept `--log-level` (e.g., `--log-level DEBUG`) to control verbosity. The default is `INFO`.

**Scan (images + videos):**
```bash
facefind-detect \
  --input /PATH/TO/MEDIA \
  --output ./outputs \
  --video-step 5 \
  --strictness strict
```

**Verify crops (reject low-quality / non-faces):**
```bash
facefind-verify \
  outputs/crops \
  --reject-dir outputs/rejects \
  --strictness strict
```

**Split a clustering/prediction CSV into per-label folders (optional):**
```bash
facefind-split \
  outputs/clusters.csv \
  outputs/people_by_cluster \
  --copy
```

**Train a classifier on labeled folders:**
```bash
facefind-train \
  --data outputs/people_by_cluster \
  --out models
```

**Predict on new crops / folders:**
```bash
facefind-predict \
  outputs/crops \
  --model-dir models \
  --out outputs/predictions.csv
```

**Apply predictions to accept/review folders (optional):**
```bash
facefind-apply \
  --in outputs/predictions.csv \
  --people-dir outputs/people \
  --out-dir outputs/sorted
```

---

## Strictness Profiles (centralized in `facefind/config.py`)

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
Tiny tests ensure config integrity and basic environment sanity. Run them from the
repository root with [pytest](https://docs.pytest.org/):

```bash
pip install pytest
pytest
```

### Repo Layout
- `facefind/`: package with CLI entry points and shared modules.
  - `utils.py`: small reusable helpers like `ensure_dir`.
  - `file_exts.py`: shared image and video file extension sets.
- `models/`: trained classifier artifacts.
- `outputs/`: crops, manifests, clusters, predictions, etc.
- `tests/`: small `pytest` suite.

---

## License
MIT © 2025 <Your Name or Organization>
See [`LICENSE`](LICENSE).
