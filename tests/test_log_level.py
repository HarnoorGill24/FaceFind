import sys
import logging
import facefind.split_clusters as split_clusters


def test_log_level_debug_enables_debug_logs(tmp_path, monkeypatch, caplog):
    img = tmp_path / "img.jpg"
    img.touch()
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(f"path,label\n{img},p1\n")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["split_clusters", str(csv_path), str(out_dir), "--log-level", "DEBUG"])
    split_clusters.main()
    logging.getLogger().addHandler(caplog.handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger(__name__).debug("debug message")
    assert "debug message" in caplog.text
    logging.getLogger().removeHandler(caplog.handler)
    logging.basicConfig(level=logging.WARNING, force=True)
