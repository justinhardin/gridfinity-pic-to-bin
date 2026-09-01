"""Console-script shim for ``pic-to-bin-web``.

The web dependencies are core, so this normally just forwards to
``server.cli()``. It exists for the case where they are missing anyway — a
partial install, a hand-pruned venv, or an upgrade from a release where they
lived in a ``[web]`` extra — because the failure otherwise arrives as a
traceback from deep inside ``server.py``. This module imports nothing from
the web stack, so it can say what to do instead.
"""

from __future__ import annotations

# Importable names behind the web dependencies — used to tell "this install
# is missing them" apart from a genuine broken import inside server.py,
# which must not be swallowed.
_WEB_MODULES = frozenset({
    "fastapi",        # fastapi
    "uvicorn",        # uvicorn[standard]
    "multipart",      # python-multipart
    "sse_starlette",  # sse-starlette
    "starlette",      # pulled in by fastapi
    "dotenv",         # python-dotenv
})

_INSTALL_HELP = """\
pic-to-bin-web could not import its web dependencies{missing}.

They ship with the package, so this points at a partial or outdated install —
older releases kept them in a separate [web] extra. Reinstalling repairs it:

    pipx install --force gridfinity-pic-to-bin           # pipx
    pip install --force-reinstall gridfinity-pic-to-bin  # pip / venv
    pip install -e .                                     # source checkout

The other commands (pic-to-bin, generate-phone-template, …) are unaffected.\
"""


def main() -> None:
    """Entry point for ``pic-to-bin-web``."""
    try:
        from pic_to_bin.web.server import cli
    except ImportError as exc:
        name = (getattr(exc, "name", "") or "").split(".")[0]
        if name not in _WEB_MODULES:
            raise  # a real import bug — show the traceback
        raise SystemExit(_INSTALL_HELP.format(missing=f" (missing: {name})")) from exc
    cli()
