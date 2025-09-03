import csv
import sys

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
