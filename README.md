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
  --config-profile strict
```

**Verify crops (reject low-quality / non-faces):**
```bash
facefind-verify \
  --input outputs/crops \
  --reject-dir outputs/rejects \
  --config-profile strict
```

**Split a clustering/prediction CSV into per-label folders (optional):**
```bash
facefind-split \
  --input outputs/clusters.csv \
  --output outputs/people_by_cluster \
  --copy
```

**Train a classifier on labeled folders:**
```bash
facefind-train \
  --input outputs/people_by_cluster \
  --models-dir models
```

**Predict on new crops / folders:**
```bash
facefind-predict \
  --input outputs/crops \
  --models-dir models \
  --output outputs/predictions.csv
```

**Apply predictions to accept/review folders (optional):**
```bash
facefind-apply \
  --input outputs/predictions.csv \
  --people-dir outputs/people \
  --output outputs/sorted
```

---

## Strictness Profiles (centralized in `facefind/config.py`)

Use a single flag everywhere: `--config-profile {strict,normal,loose}`.

| Profile | MTCNN Thresholds (pnet / rnet / onet) | Min Face Size | Min Prob | Embed Batch |
|---|---|---|---|---|
| strict | 0.80 / 0.90 / 0.95 | 60 px | 0.95 | 64 |
| normal | 0.70 / 0.80 / 0.92 | 40 px | 0.90 | 96 |
| loose  | 0.60 / 0.70 / 0.90 | 32 px | 0.85 | 128 |

> These defaults balance recall vs precision and memory usage. Override per-script with explicit flags if you need to.

---

## Troubleshooting

- **False positives (e.g., crops of shirts):** use `--config-profile strict` or raise `--min-prob 0.95` and `--min-size 60`.
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
- `utils/`: reusable helpers like `ensure_dir` and `IMAGE_EXTS`.
- `facefind/`: package with CLI entry points and higher level utilities.
  - `utils.py`: helpers such as `is_image` and `sanitize_label`.
  - `file_exts.py`: shared video file extension set.
- `models/`: trained classifier artifacts.
- `outputs/`: crops, manifests, clusters, predictions, etc.
- `tests/`: small `pytest` suite.

---

## License

This project is licensed under the [MIT License](LICENSE).
