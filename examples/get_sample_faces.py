#!/usr/bin/env python3
"""Download a couple of synthetic face images for quick tests.

Images come from https://thispersondoesnotexist.com and do **not** depict real people.
"""

import urllib.request
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).parent / "sample_media"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        url = "https://thispersondoesnotexist.com/"
        dest = out_dir / f"face{i + 1}.jpg"
        with urllib.request.urlopen(url) as resp:
            dest.write_bytes(resp.read())
        print(f"Wrote {dest}")
    print("Done. Use examples/sample_media for quick experiments.")


if __name__ == "__main__":
    main()
