#!/usr/bin/env python3
"""
embedding_utils.py
Shared utilities for device selection, image loading, and face embeddings.
Used by: main.py, train_face_classifier.py, predict_face.py
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, TypeVar


T = TypeVar("T")


# -------------------------
# Device selection
# -------------------------
def get_device(preferred: Optional[str] = "auto") -> str:
    """
    Decide which torch device to use.
    Honors a user-preferred value if available; otherwise auto-selects.
    """
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ImportError("torch is required for get_device()") from e

    pref = (preferred or "auto").lower()
    allowed = {"auto", "cpu", "cuda", "mps"}
    if pref not in allowed:
        raise ValueError(f"Unknown device '{preferred}'. Allowed: {sorted(allowed)}")

    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if pref == "cpu":
        return "cpu"

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------
# Model (cached per device)
# -------------------------
@functools.lru_cache(maxsize=2)
def _get_embed_model(device: str) -> InceptionResnetV1:
    """
    Load Facenet InceptionResnetV1 once per device and cache it.
    """
    try:
        from facenet_pytorch import InceptionResnetV1
    except ModuleNotFoundError as e:
        raise ImportError("facenet_pytorch is required for embeddings") from e

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


# -------------------------
# Image I/O
# -------------------------
def load_images(paths: Sequence[Path]) -> List[Optional[Image.Image]]:
    """
    Load images as PIL (RGB). On failure returns None for that slot.
    Preserves one-to-one alignment with `paths`.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as e:
        raise ImportError("Pillow is required for load_images()") from e

    out: List[Optional[Image.Image]] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                out.append(im.convert("RGB"))
        except Exception as exc:
            logging.warning("Failed to load image %s: %s", p, exc)
            out.append(None)
    return out


# -------------------------
# Preprocess → Tensor
# -------------------------
def _preprocess(im: Image.Image) -> torch.Tensor:
    """
    Facenet expects 160x160 RGB, float in [-1,1].
    Returns CHW float32 tensor.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as e:
        raise ImportError("Pillow is required for preprocessing") from e
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise ImportError("NumPy is required for preprocessing") from e
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ImportError("torch is required for preprocessing") from e

    im2 = im.resize((160, 160), Image.BILINEAR)
    # Make a writable copy to avoid the PyTorch "non-writable NumPy array" warning.
    arr = np.asarray(im2, dtype=np.float32).copy()  # HWC, float32 in [0,255]
    arr = arr / 255.0  # [0,1]
    arr = (arr - 0.5) / 0.5  # [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return t


# -------------------------
# Batch embeddings
# -------------------------
def embed_images(
    imgs: Sequence[Optional[Image.Image]],
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute embeddings for a list of PIL images (some entries may be None).
    Returns an array of shape (len(imgs), 512). For images that failed to load,
    a zero vector is returned at that index to preserve alignment with labels.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise ImportError("NumPy is required for embed_images()") from e
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ImportError("torch is required for embed_images()") from e

    dev = get_device(device)
    model = _get_embed_model(dev)

    valid_ix: List[int] = []
    batch_tensors: List[torch.Tensor] = []

    for i, im in enumerate(imgs):
        if im is None:
            continue
        batch_tensors.append(_preprocess(im))
        valid_ix.append(i)

    # Nothing valid → return all zeros
    d = 512
    if not valid_ix:
        return np.zeros((len(imgs), d), dtype=np.float32)

    # Compute embeddings for valid images
    embs_valid = np.zeros((len(valid_ix), d), dtype=np.float32)
    with torch.no_grad():
        if dev == "mps":
            # Allow CPU fallback for kernels missing on MPS (safer on Apple Silicon).
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        for start in range(0, len(valid_ix), batch_size):
            chunk = batch_tensors[start : start + batch_size]
            batch = torch.stack(chunk).to(dev, non_blocking=True)
            out = model(batch).cpu().numpy().astype(np.float32, copy=False)
            embs_valid[start : start + out.shape[0]] = out

    # Place into full array aligned to original indices, fill None-slots with zeros
    embs = np.zeros((len(imgs), d), dtype=np.float32)
    for dst_row, src_row in enumerate(valid_ix):
        embs[src_row] = embs_valid[dst_row]

    return embs


# -------------------------
# General helpers
# -------------------------
def batched(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Yield successive chunks (lists) of up to ``size`` items from iterable.

    Works with any iterable. The final chunk may be shorter.
    Example: list(batched(range(5), 2)) -> [[0, 1], [2, 3], [4]]
    """
    if size <= 0:
        raise ValueError("size must be positive")

    it = iter(iterable)
    while True:
        chunk: List[T] = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        else:
            yield chunk
