import csv
import sys

from facefind import apply_predictions


def test_detect_headers_synonyms():
    headers = ["Image", "Pred_Label", "Confidence"]
    assert apply_predictions.detect_headers(headers) == (
        "Image",
        "Pred_Label",
        "Confidence",
    )


def test_apply_predictions_label_sanitization_and_placement(tmp_path, monkeypatch):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img3 = tmp_path / "c.jpg"
    for img, text in [(img1, "a"), (img2, "b"), (img3, "c")]:
        img.write_text(text)

    csv_path = tmp_path / "preds.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "prediction", "score"])
        writer.writerow(["a.jpg", "al/ice", "0.9"])
        writer.writerow(["b.jpg", "bo b", "0.6"])
        writer.writerow(["c.jpg", "ca r/ol", "0.2"])

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

    # high confidence → accept
    assert (out_dir / "accept" / "al_ice" / "a.jpg").exists()

    # mid confidence → review
    assert (out_dir / "review" / "bo_b" / "b.jpg").exists()

    # low confidence → ignored
    assert not (out_dir / "accept" / "ca_r_ol" / "c.jpg").exists()
    assert not (out_dir / "review" / "ca_r_ol" / "c.jpg").exists()

