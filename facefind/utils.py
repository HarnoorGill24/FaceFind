"""Shared utility helpers for FaceFind scripts.

This module centralizes small helpers used across multiple scripts:

* :func:`is_image` – quick predicate for image paths.
* :func:`sanitize_label` – normalize labels for filesystem safety.
* :func:`ensure_dir` – create directories as needed.

The canonical file-extension set is defined here to avoid heavy imports at CLI
startup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    """Ensure directory *p* exists, creating parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


__all__ = ["IMAGE_EXTS", "ensure_dir", "is_image", "sanitize_label"]


def is_image(p: Path) -> bool:
    """Return True if *p* has an image file extension."""
    return p.suffix.lower() in IMAGE_EXTS


def sanitize_label(
    label: str,
    replacement: str | None = "_",
    max_length: int = 100,
) -> str:
    """Normalize *label* for safe filesystem usage.

    Parameters
    ----------
    label:
        Raw label to clean.
    replacement:
        String used to substitute disallowed characters. ``None`` strips
        those characters instead of replacing them. Defaults to ``"_"``.
    max_length:
        Maximum length for the sanitized label. Longer inputs are truncated.
        Set to ``0`` to disable the limit. Defaults to ``100``.
    """

    label = (label or "").strip()
    if not label:
        return "unknown"

    # Remove any path traversal components ("..") first
    label = label.replace("..", "")

    # Avoid path separators entirely
    for sep in {os.sep, os.altsep}:
        if sep:
            label = label.replace(sep, replacement or "")

    # Permit only alphanumeric characters plus -_
    pattern = r"[^A-Za-z0-9_-]"
    if replacement is not None:
        label = re.sub(pattern, replacement, label)
    else:
        label = re.sub(pattern, "", label)

    label = label.strip()
    if replacement:
        label = label.strip(replacement)

    if not label:
        return "unknown"

    if max_length:
        label = label[:max_length]

    return label or "unknown"
