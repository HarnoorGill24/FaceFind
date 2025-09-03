"""Shared utility helpers for FaceFind scripts.

This module centralizes small helpers used across multiple scripts:

* :data:`IMAGE_EXTS` – set of supported image file extensions.
* :func:`is_image` – quick predicate for image paths.
* :func:`ensure_dir` – create a directory tree if it doesn't exist.

Import these helpers instead of redefining them in each script so that
future tools stay consistent.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Common image file extensions supported by FaceFind
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    """Return True if *p* has an image file extension."""
    return p.suffix.lower() in IMAGE_EXTS


def ensure_dir(p: Path) -> None:
    """Ensure directory *p* exists, creating parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str, replacement: str | None = "_") -> str:
    """Normalize *label* for safe filesystem usage.

    Parameters
    ----------
    label:
        Raw label to clean.
    replacement:
        String used to substitute disallowed characters. ``None`` strips
        those characters instead of replacing them. Defaults to ``"_"``.
    """

    label = (label or "").strip()
    if not label:
        return "unknown"

    # Avoid path traversal / separators
    for sep in {os.sep, os.altsep}:
        if sep:
            label = label.replace(sep, replacement or "")

    # Optionally clean up any remaining non-alphanumeric characters
    if replacement is not None:
        label = re.sub(r"[^\w.-]", replacement, label)
    else:
        label = re.sub(r"[^\w.-]", "", label)

    label = label.strip()
    return label or "unknown"

