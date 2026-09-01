"""Packaging invariants that only bite after an install.

The failure mode these cover is invisible in a source checkout: a console
script that is registered but cannot import its dependencies, or a
documented install command that stops resolving.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def project():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _names(specs):
    """Requirement strings -> bare distribution names."""
    return {s.split("[")[0].split(">")[0].split("=")[0].strip() for s in specs}


def test_web_app_dependencies_are_core(project):
    """`pipx install gridfinity-pic-to-bin` must serve the web app.

    These lived in a [web] extra, which left a plain install shipping a
    pic-to-bin-web script that died on `import fastapi`.
    """
    core = _names(project["dependencies"])
    for dist in ("fastapi", "uvicorn", "python-multipart", "sse-starlette"):
        assert dist in core, f"{dist} must be a core dependency"


def test_web_extra_still_resolves(project):
    """Older docs and muscle memory say [web]; it must not become an error."""
    extras = project["optional-dependencies"]
    assert "web" in extras
    assert _names(extras["web"]) == _names(extras["llm"]) == {"anthropic"}


def test_web_console_script_goes_through_the_launcher(project):
    """server:cli imports FastAPI at module scope; launcher:main does not."""
    assert project["scripts"]["pic-to-bin-web"] == "pic_to_bin.web.launcher:main"
