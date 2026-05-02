"""Vendor Lit locally so the web app has no third-party runtime dependency.

Usage::

    python -m pic_to_bin.web.vendor_lit

Downloads ``lit-all`` from a CDN into ``pic_to_bin/web/static/`` and rewrites
the import map in ``index.html`` to point at the local copy. Idempotent —
running it again just refreshes the file at the pinned version.

If your machine cannot reach the CDN, manually save ``lit-all.min.js`` from
https://cdn.jsdelivr.net/npm/lit@3.2.1/+esm into the static dir and re-run
this script — it will detect the existing file and just rewrite the import
map.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

LIT_VERSION = "3.2.1"
LIT_URL = f"https://cdn.jsdelivr.net/npm/lit@{LIT_VERSION}/+esm"
DECORATORS_URL = f"https://cdn.jsdelivr.net/npm/lit@{LIT_VERSION}/decorators.js/+esm"

STATIC_DIR = Path(__file__).parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"


def _download(url: str, target: Path) -> None:
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        target.write_bytes(resp.read())
    print(f"  wrote {target} ({target.stat().st_size:,} bytes)")


def _rewrite_importmap(local: bool) -> None:
    text = INDEX_PATH.read_text(encoding="utf-8")
    if local:
        new_imports = (
            '"lit": "/static/lit-all.min.js",\n'
            '        "lit/decorators.js": "/static/lit-decorators.js"'
        )
    else:
        new_imports = (
            f'"lit": "https://esm.sh/lit@{LIT_VERSION}",\n'
            f'        "lit/decorators.js": "https://esm.sh/lit@{LIT_VERSION}/decorators.js"'
        )
    pattern = re.compile(
        r'"lit":\s*"[^"]*",\s*\n\s*"lit/decorators\.js":\s*"[^"]*"',
        re.MULTILINE,
    )
    new_text, n = pattern.subn(new_imports, text)
    if n == 0:
        print("WARN: import map block not found in index.html — leaving file unchanged.")
        return
    INDEX_PATH.write_text(new_text, encoding="utf-8")
    print(f"  updated import map in {INDEX_PATH} -> {'local' if local else 'CDN'}")


def main() -> int:
    target_lit = STATIC_DIR / "lit-all.min.js"
    target_decorators = STATIC_DIR / "lit-decorators.js"
    try:
        _download(LIT_URL, target_lit)
        _download(DECORATORS_URL, target_decorators)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR downloading Lit: {e}", file=sys.stderr)
        print(
            "Save the files manually from\n"
            f"  {LIT_URL}\n"
            f"  {DECORATORS_URL}\n"
            f"into {STATIC_DIR} and re-run this script.",
            file=sys.stderr,
        )
        return 1
    _rewrite_importmap(local=True)
    print("\nDone. Restart the server (or just reload the browser).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
