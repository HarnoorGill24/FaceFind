import os
import shutil
from pathlib import Path

from utils.common import ensure_dir


def unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suf = dst.stem, dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}-{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> Path:
    dst = unique_path(dst)
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)
    return dst
