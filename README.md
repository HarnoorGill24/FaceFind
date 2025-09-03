# FaceFind

End-to-end local pipeline for **face discovery and labeling** across large photo/video libraries.

**Pipeline:** detect → verify → cluster → train classifier → predict → apply.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

---

## Quickstart

Grab a few images or short videos. A tiny synthetic set can be fetched via:

```bash
python examples/get_sample_faces.py  # downloads into examples/sample_media/
```

Then run a minimal flow:

```bash
facefind-detect --input examples/sample_media --output outputs --config-profile strict
# label/cluster crops under outputs/crops/clustered before training
facefind-train --input outputs/crops/clustered --models-dir models
facefind-predict --input some/new/images --models-dir models --output outputs/predictions.csv
facefind-apply --input outputs/predictions.csv --output outputs/sorted
```

Accepted images land in `outputs/sorted/accept` and lower-confidence ones in `outputs/sorted/review`.

---

## CLI Reference

- `facefind-detect` – scan media and save face crops + manifest.
- `facefind-verify` – filter out low-quality or non-face crops.
- `facefind-split` – break a clustering/prediction CSV into per-label folders.
- `facefind-train` – train a classifier from labeled image folders.
- `facefind-predict` – predict labels for new crops or folders.
- `facefind-apply` – organize images into accept/review trees based on confidence.

---

## Repo Layout

```
.
├─ facefind/          # library code (no side effects on import)
├─ tests/             # unit & integration tests
├─ examples/          # tiny demo inputs or generator
├─ models/            # (gitignored) trained artifacts
├─ outputs/           # (gitignored) run outputs
├─ scripts/           # sanity scripts (optional)
├─ docs/              # design & usage docs
└─ pyproject.toml     # console scripts, tool configs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline details.

---

## Strictness Profiles (centralized in `facefind/config.py`)

Use a single flag everywhere: `--config-profile {strict,normal,loose}`.

| Profile | MTCNN Thresholds (pnet / rnet / onet) | Min Face Size | Min Prob | Embed Batch |
|---|---|---|---|---|
| strict | 0.80 / 0.90 / 0.95 | 60 px | 0.95 | 64 |
| normal | 0.70 / 0.80 / 0.92 | 40 px | 0.90 | 96 |
| loose  | 0.60 / 0.70 / 0.90 | 32 px | 0.85 | 128 |

> These defaults balance recall vs precision and memory usage. Override per-script with explicit flags if needed.

---

## Troubleshooting

- **False positives (e.g., crops of shirts):** use `--config-profile strict` or raise `--min-prob 0.95` and `--min-size 60`.
- **Killed with exit code 137 (OOM):** reduce dataset / run in shards, lower `--embed-batch` (e.g., 32), or try `strict` profile.
- **kNN CV error with tiny classes:** ensure ≥3 samples per class or favor SVM fallback.
- **Slow detection:** switch hardware backend to GPU/MPS if available; RetinaFace/InsightFace planned.

---

## Development

### Tests
Tiny tests ensure config integrity and basic environment sanity. Run them from the repository root with [pytest](https://docs.pytest.org/):

```bash
pip install pytest
pytest
```

---

## License

This project is licensed under the [MIT License](LICENSE).
