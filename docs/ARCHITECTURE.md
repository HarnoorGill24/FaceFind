# Architecture

This document sketches the flow behind FaceFind's command line tools. See the
[README](../README.md) for a high level overview.

```
images/videos
    |
    v
facefind-detect  --> crops_manifest.csv + crops/
    |
    v
(facefind-verify) [optional] --> verified crops
    |
    v
facefind-split   --> clustered & labeled folders
    |
    v
facefind-train   --> models/
    |
    v
facefind-predict --> predictions.csv
    |
    v
facefind-apply   --> outputs/sorted
```

## Console scripts

| CLI script        | Internal entry point                     |
|-------------------|------------------------------------------|
| `facefind-detect` | `facefind.main:main`                     |
| `facefind-verify` | `facefind.verify_crops:main`             |
| `facefind-split`  | `facefind.split_clusters:main`           |
| `facefind-train`  | `facefind.train_face_classifier:main`    |
| `facefind-predict`| `facefind.predict_face:main`             |
| `facefind-apply`  | `facefind.apply_predictions:main`        |
| `facefind-report` | `facefind.report:main`                   |

## Configuration

Strictness profiles and shared thresholds live in
[`facefind/config.py`](../facefind/config.py). Each CLI adds
`--config-profile` via helpers in [`facefind/cli_common.py`](../facefind/cli_common.py)
and resolves it with `config.get_profile`, which raises on unknown names. File
system paths are validated early with `cli_common.validate_path` to fail fast on
missing inputs.
