import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from config import get_profile

@pytest.mark.parametrize(
    "name, thresholds",
    [
        ("strict", [0.80, 0.90, 0.95]),
        ("normal", [0.70, 0.80, 0.92]),
        ("loose", [0.60, 0.70, 0.90]),
    ],
)
def test_get_profile_thresholds(name, thresholds):
    profile = get_profile(name)
    assert profile.mtcnn_thresholds == thresholds
