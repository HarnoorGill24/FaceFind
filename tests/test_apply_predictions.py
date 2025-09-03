import csv
import sys

import pytest

from facefind import apply_predictions


def test_apply_predictions_header_and_placement(tmp_path, monkeypatch):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img3 = tmp_path / "c.jpg"
    for img, text in [(img1, "a"), (img2, "b"), (img3, "c")]:
        img.write_text(text)
    csv_path = tmp_path / "preds.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "prediction", "score"])
        writer.writerow(["a.jpg", "alice", "0.9"])
        writer.writerow(["b.jpg", "bob", "0.6"])
        writer.writerow(["c.jpg", "carol", "0.2"])
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = apply_predictions.detect_headers(reader.fieldnames)
    assert cols == ("file", "prediction", "score")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_predictions",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--rel-root",
            str(tmp_path),
        ],
    )
    apply_predictions.main()
    assert (out_dir / "accept" / "alice" / "a.jpg").exists()
    assert (out_dir / "review" / "bob" / "b.jpg").exists()
    assert not (out_dir / "accept" / "carol" / "c.jpg").exists()
    assert not (out_dir / "review" / "carol" / "c.jpg").exists()


def _write_header_only_csv(path):
    path.write_text("path,label,prob\n")


def test_threshold_rejection(tmp_path, monkeypatch):
    csv_path = tmp_path / "preds.csv"
    _write_header_only_csv(csv_path)
    out_dir = tmp_path / "out"
    for opt, val in [
        ("--accept-threshold", "-0.1"),
        ("--accept-threshold", "1.1"),
        ("--review-threshold", "-0.1"),
        ("--review-threshold", "1.1"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            ["apply_predictions", str(csv_path), "--out-dir", str(out_dir), opt, val],
        )
        with pytest.raises(SystemExit) as exc:
            apply_predictions.main()
        assert opt.lstrip("-") in str(exc.value)


@pytest.mark.parametrize("thresh", [0.0, 1.0])
def test_threshold_boundaries(tmp_path, monkeypatch, thresh):
    csv_path = tmp_path / "preds.csv"
    _write_header_only_csv(csv_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_predictions",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--accept-threshold",
            str(thresh),
            "--review-threshold",
            str(thresh),
        ],
    )
    apply_predictions.main()
    assert (out_dir / "accept").exists()
