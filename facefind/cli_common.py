from __future__ import annotations

import argparse
from pathlib import Path

from facefind import __version__

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
DEVICES = ["auto", "cpu", "mps", "cuda"]
PROFILES = ["strict", "normal", "loose"]


def add_version(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="version", version=f"facefind {__version__}")


def add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        type=str.upper,
        help="Set logging level",
    )


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="auto",
        choices=DEVICES,
        type=str.lower,
        help="Computation device",
    )


def add_config_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-profile",
        default="strict",
        choices=PROFILES,
        type=str.lower,
        help="Configuration profile",
    )


def validate_path(path: Path, *, kind: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{kind} path does not exist: {path}")
    return path
