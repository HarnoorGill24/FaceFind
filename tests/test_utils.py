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


def test_sanitize_label_empty():
    assert sanitize_label("") == "unknown"


def test_sanitize_label_path_traversal():
    assert sanitize_label("../secret") == "secret"


def test_sanitize_label_special_chars_and_dots():
    label = "foo.bar?baz"
    assert sanitize_label(label) == "foo_bar_baz"


def test_sanitize_label_max_length():
    long_label = "a" * 150
    assert len(sanitize_label(long_label)) == 100
