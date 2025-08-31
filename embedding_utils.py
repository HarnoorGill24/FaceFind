from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1


def get_device(device: Optional[str] = None) -> str:
    """Select an available torch device.

    If *device* is provided, it is returned as-is. Otherwise the function
    prefers CUDA, then MPS, and finally CPU.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batched(iterable: Iterable, n: int):
    """Yield lists of size *n* from *iterable*."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def load_images(paths: List[Path]) -> List[Optional[Image.Image]]:
    """Load images from *paths* returning ``None`` for failures."""
    out: List[Optional[Image.Image]] = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception:
            out.append(None)
    return out


def embed_images(
    imgs: List[Optional[Image.Image]],
    batch_size: int,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute FaceNet embeddings for *imgs*.

    ``None`` entries are skipped but still reserve a row in the output array.
    Images are resized to 160x160 RGB and processed in batches. The returned
    array has shape ``(N, 512)`` where ``N`` is ``len(imgs)``.
    """
    device = get_device(device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    embs = np.zeros((len(imgs), 512), dtype=np.float32)

    def preprocess(pil: Image.Image) -> Image.Image:
        return pil.resize((160, 160))

    idx = 0
    for chunk in batched(imgs, batch_size):
        good_idx: List[int] = []
        tensors = []
        for j, im in enumerate(chunk):
            if im is None:
                continue
            try:
                im2 = preprocess(im)
                t = torch.from_numpy(np.asarray(im2)).permute(2, 0, 1).float() / 255.0
                tensors.append(t.unsqueeze(0))
                good_idx.append(j)
            except Exception:
                pass
        if not tensors:
            idx += len(chunk)
            continue
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = resnet(batch).cpu().numpy().astype(np.float32)
        for j_local, vec in zip(good_idx, feats):
            embs[idx + j_local, :] = vec
        idx += len(chunk)
    return embs
