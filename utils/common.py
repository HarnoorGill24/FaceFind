from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    """Ensure directory *p* exists, creating parents if needed."""
    p.mkdir(parents=True, exist_ok=True)
