"""Shared utility helpers for FaceFind scripts.

This module centralizes small helpers used across multiple scripts:

* :data:`IMAGE_EXTS` – set of supported image file extensions.
* :func:`is_image` – quick predicate for image paths.
* :func:`ensure_dir` – create a directory tree if it doesn't exist.

Import these helpers instead of redefining them in each script so that
future tools stay consistent.
"""
from __future__ import annotations

from pathlib import Path
import os

# Common image file extensions supported by FaceFind
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    """Return True if *p* has an image file extension."""
    return p.suffix.lower() in IMAGE_EXTS


def ensure_dir(p: Path) -> None:
    """Ensure directory *p* exists, creating parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str) -> str:
    """Normalize *label* for safe filesystem usage."""
    label = (label or "").strip()
    if not label:
        return "unknown"
    # Avoid path traversal / separators
    return label.replace(os.sep, "_")

