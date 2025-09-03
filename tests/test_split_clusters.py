import csv
import sys
import tracemalloc

from facefind import split_clusters


def test_split_clusters_streams_large_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "big.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "cluster"])
        for i in range(200_000):
            writer.writerow([f"/nope/{i}.jpg", "x"])

    out_dir = tmp_path / "out"

    tracemalloc.start()
    monkeypatch.setattr(
        sys, "argv", ["split_clusters", "--input", str(csv_path), "--output", str(out_dir)]
    )
    split_clusters.main()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 25 * 1024 * 1024  # peak memory stays under 25MB


def test_split_clusters_places_files(tmp_path, monkeypatch):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img1.write_text("a")
    img2.write_text("b")
    csv_path = tmp_path / "small.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction"])
        writer.writerow(["a.jpg", "foo"])
        writer.writerow(["b.jpg", "bar"])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_clusters",
            "--input",
            str(csv_path),
            "--output",
            str(out_dir),
            "--rel-root",
            str(tmp_path),
        ],
    )
    split_clusters.main()
    assert (out_dir / "foo" / "a.jpg").exists()
    assert (out_dir / "bar" / "b.jpg").exists()
