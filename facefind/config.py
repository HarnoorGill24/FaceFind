"""Configuration profiles for FaceFind detection and embedding."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StrictnessProfile:
    """
    Centralized thresholds for detector + filtering + batching.
    Keep these in sync for a consistent pipeline across scripts.
    """

    name: str
    mtcnn_thresholds: List[float]  # [pnet, rnet, onet] probabilities in (0, 1]
    min_size: int  # minimum face bounding box (px)
    min_prob: float  # minimum face probability to keep crop
    embed_batch: int  # embedding batch size (lower to save RAM)


# Tune once; used everywhere via get_profile()
STRICTNESS: Dict[str, StrictnessProfile] = {
    "strict": StrictnessProfile(
        name="strict",
        mtcnn_thresholds=[0.80, 0.90, 0.95],
        min_size=60,
        min_prob=0.95,
        embed_batch=64,
    ),
    "normal": StrictnessProfile(
        name="normal",
        mtcnn_thresholds=[0.70, 0.80, 0.92],
        min_size=40,
        min_prob=0.90,
        embed_batch=96,
    ),
    "loose": StrictnessProfile(
        name="loose",
        mtcnn_thresholds=[0.60, 0.70, 0.90],
        min_size=32,
        min_prob=0.85,
        embed_batch=128,
    ),
}


def get_profile(name: str) -> StrictnessProfile:
    """
    Resolve a named strictness profile.
    Raises ValueError for unknown names to fail fast.
    """
    key = (name or "strict").strip().lower()
    if key not in STRICTNESS:
        raise ValueError(f"Unknown strictness '{name}'. Use one of: {', '.join(STRICTNESS)}")
    return STRICTNESS[key]
