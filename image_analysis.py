#!/usr/bin/env python3
"""
image_analysis.py

High-level utilities for image quality scoring, captioning/tagging,
object detection, OCR, and duplicate search.

The functions lazily import heavy dependencies so that the module can be
imported even if optional packages (e.g. transformers, ultralytics) are
not installed.  Each routine attempts to use the best available torch
backend, including Apple's MPS backend on Apple Silicon, via
``embedding_utils.get_device``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from embedding_utils import get_device


# ---------------------------------------------------------------------------
# Technical quality scoring
# ---------------------------------------------------------------------------

def variance_of_laplacian(img: np.ndarray) -> float:
    """Return the variance of the Laplacian -- a fast sharpness metric."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_metrics(img: np.ndarray) -> Dict[str, float]:
    """Compute simple under/over exposure fractions from a grayscale histogram."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum() or 1.0
    return {
        "dark_frac": float(hist[:30].sum() / total),  # under-exposed
        "bright_frac": float(hist[225:].sum() / total),  # over-exposed
    }


def face_region_sharpness(img: np.ndarray, device: Optional[str] = None) -> List[float]:
    """Compute sharpness for each detected face region using MTCNN."""
    from facenet_pytorch import MTCNN  # lazy import

    dev = get_device(device)
    detector = MTCNN(keep_all=True, device=dev)
    boxes, _ = detector.detect(Image.fromarray(img))
    if boxes is None:
        return []
    sharpness = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = img[y1:y2, x1:x2]
        sharpness.append(variance_of_laplacian(crop))
    return sharpness


def face_is_usable(img: np.ndarray, device: Optional[str] = None, *, min_sharpness: float = 100.0) -> bool:
    """Return True if any detected face exceeds ``min_sharpness``."""
    vals = face_region_sharpness(img, device=device)
    return bool(vals) and max(vals) >= min_sharpness


# ---------------------------------------------------------------------------
# Auto captioning & tagging (no training)
# ---------------------------------------------------------------------------

def generate_caption(image: Image.Image, device: Optional[str] = None) -> str:
    """Generate a plain-English caption using BLIP."""
    from transformers import BlipForConditionalGeneration, BlipProcessor  # lazy

    dev = get_device(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(dev)
    inputs = processor(images=image, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def zero_shot_tags(
    image: Image.Image,
    labels: Sequence[str],
    device: Optional[str] = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k labels using CLIP zero-shot classification."""
    from transformers import CLIPModel, CLIPProcessor  # lazy

    dev = get_device(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
    inputs = processor(text=list(labels), images=image, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy().ravel()
    pairs = list(zip(labels, logits.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Object detection (e.g., soccer context)
# ---------------------------------------------------------------------------

def detect_objects(
    img: np.ndarray,
    model_name: str = "yolov8n.pt",
    device: Optional[str] = None,
):
    """Run YOLOv8/11 via the ultralytics package and return the first result."""
    from ultralytics import YOLO  # lazy

    dev = get_device(device)
    model = YOLO(model_name)
    results = model.predict(img, device=dev, verbose=False)
    return results[0]


# ---------------------------------------------------------------------------
# OCR (e.g., jersey numbers, signage)
# ---------------------------------------------------------------------------

def ocr_text(img: np.ndarray, device: Optional[str] = None) -> List[str]:
    """Extract text using PaddleOCR. Returns a list of strings."""
    from paddleocr import PaddleOCR  # lazy

    dev = get_device(device)
    use_gpu = dev in {"cuda", "mps"}
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu, show_log=False)
    result = ocr.ocr(img, cls=True)
    texts: List[str] = []
    for line in result:
        for _, (text, _conf) in line:
            texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Near-duplicate & similarity search
# ---------------------------------------------------------------------------

def clip_embeddings(images: Sequence[Image.Image], device: Optional[str] = None) -> np.ndarray:
    """Return L2-normalized CLIP embeddings for a list of PIL images."""
    from transformers import CLIPModel, CLIPProcessor  # lazy

    dev = get_device(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
    feats: List[np.ndarray] = []
    for im in images:
        inputs = processor(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            feat = model.get_image_features(**inputs).cpu().numpy().ravel()
        feats.append(feat / np.linalg.norm(feat))
    return np.vstack(feats)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index (inner product) for similarity search."""
    import faiss  # lazy

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def perceptual_hash(image: Image.Image) -> str:
    """Return a pHash string for quick duplicate pre-filtering."""
    import imagehash  # lazy

    return str(imagehash.phash(image))


def find_near_duplicates(
    images: Sequence[Image.Image],
    device: Optional[str] = None,
    thresh: float = 0.95,
) -> Dict[int, List[int]]:
    """Return a mapping of image index -> list of near-duplicate indices."""
    embs = clip_embeddings(images, device=device)
    index = build_faiss_index(embs)
    sims, idxs = index.search(embs, k=embs.shape[0])
    groups: Dict[int, List[int]] = {}
    for i, (sim_row, idx_row) in enumerate(zip(sims, idxs)):
        dup = [j for s, j in zip(sim_row[1:], idx_row[1:]) if s >= thresh]
        if dup:
            groups[i] = dup
    return groups
