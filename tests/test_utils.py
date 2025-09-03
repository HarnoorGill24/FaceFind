import os

from facefind.utils import sanitize_label


def test_sanitize_label_replaces_separators():
    label = f"foo{os.sep}bar"
    assert sanitize_label(label) == "foo_bar"


def test_sanitize_label_replaces_altsep(monkeypatch):
    monkeypatch.setattr(os, "altsep", ";")
    label = "foo;bar"
    assert sanitize_label(label) == "foo_bar"


def test_sanitize_label_substitute_non_alnum_default():
    label = "foo!bar"
    assert sanitize_label(label) == "foo_bar"


def test_sanitize_label_strip_non_alnum():
    assert sanitize_label("foo!bar#baz", replacement=None) == "foobarbaz"
    # All stripped -> unknown
    assert sanitize_label("!!!", replacement=None) == "unknown"
