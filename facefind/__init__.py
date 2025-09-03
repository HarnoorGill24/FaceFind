"""FaceFind package with shared utilities and CLI entry points."""

__version__ = "0.1.0"

# Re-export common schema definitions for convenience when importing the
# package directly (``import facefind``).
from . import io_schema

__all__ = ["__version__", "io_schema"]

