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
    monkeypatch.setattr(sys, "argv", ["split_clusters", str(csv_path), str(out_dir)])
    split_clusters.main()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 10 * 1024 * 1024  # peak memory stays under 10MB
