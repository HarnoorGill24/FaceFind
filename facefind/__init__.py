"""FaceFind package with shared utilities and CLI entry points."""

from importlib.metadata import PackageNotFoundError, version

# Re-export common schema definitions for convenience when importing the
# package directly (``import facefind``).
from . import io_schema

try:  # pragma: no cover - importlib.metadata uses environment
    __version__ = version("facefind")
except PackageNotFoundError:  # pragma: no cover - during local editing
    __version__ = "0.0.0"

__all__ = ["io_schema", "__version__"]

