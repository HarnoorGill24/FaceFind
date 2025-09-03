import pytest

from facefind.config import get_profile


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


def test_get_profile_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        get_profile("unknown")


def test_get_profile_defaults_to_strict_for_none_and_empty():
    assert get_profile("").name == "strict"
    # type: ignore[arg-type] - exercise runtime behavior for None
    assert get_profile(None).name == "strict"  # noqa: E501
