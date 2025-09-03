"""FaceFind package with shared utilities and CLI entry points."""

# Re-export common schema definitions for convenience when importing the
# package directly (``import facefind``).
from . import io_schema

__all__ = ["io_schema"]

