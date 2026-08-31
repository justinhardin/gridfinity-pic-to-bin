"""(Re)download the vendored Lit bundle used by the web app.

Usage::

    python -m pic_to_bin.web.vendor_lit

``static/lit-all.min.js`` ships with the package, so you normally never need
this — the browser loads Lit from your own origin, which is what the server's
Content-Security-Policy allows. Run this only to restore the file if it went
missing, or to move to a new pinned ``LIT_VERSION``.

The download is the official pre-bundled build from the ``lit/dist`` repo. It
is genuinely self-contained: unlike jsdelivr's ``/npm/lit/+esm`` (a ~500 byte
stub that re-imports the real modules from the CDN at runtime), nothing in
this file reaches back out to the network.

Offline? Save the file at ``LIT_URL`` by hand into the static dir — that is
all this script does.

Note: the bundle exports Lit's core and directives but *not*
``lit/decorators.js``; ``app.js`` uses plain class fields with
``static properties`` instead of decorators.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

LIT_VERSION = "3.2.1"
LIT_URL = f"https://cdn.jsdelivr.net/gh/lit/dist@{LIT_VERSION}/all/lit-all.min.js"

STATIC_DIR = Path(__file__).parent / "static"
TARGET = STATIC_DIR / "lit-all.min.js"


def main() -> int:
    print(f"  fetching {LIT_URL}")
    try:
        with urllib.request.urlopen(LIT_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR downloading Lit: {e}", file=sys.stderr)
        print(f"Save {LIT_URL}\ninto {TARGET} by hand instead.", file=sys.stderr)
        return 1

    if b"export{" not in data:
        print(
            f"ERROR: {LIT_URL} did not return an ES module — refusing to "
            f"overwrite {TARGET}.",
            file=sys.stderr,
        )
        return 1

    # Drop the source-map comment: we don't ship the .map, and leaving the
    # pointer only earns a 404 whenever someone opens devtools.
    lines = [
        ln for ln in data.split(b"\n")
        if not ln.startswith(b"//# sourceMappingURL=")
    ]
    TARGET.write_bytes(b"\n".join(lines))
    print(f"  wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")
    print("\nDone. Reload the browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
