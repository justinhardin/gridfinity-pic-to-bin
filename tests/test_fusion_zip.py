"""Tests for the /download/fusion-addin.zip bundle.

The endpoint builds the ZIP from files on disk each call, so a rename or a
removed directory breaks it at request time only — nothing else in the suite
touches those paths. That is exactly how it regressed: the builder kept
requiring ``pic_to_bin_script/`` after the Fusion side consolidated to an
add-in-only layout, and every download 500'd.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from pic_to_bin.web import server

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def bundle(tmp_path):
    with TestClient(server.create_app(tmp_path / "jobs")) as c:
        r = c.get("/download/fusion-addin.zip")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert "pic-to-bin-fusion.zip" in r.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_bundle_holds_a_complete_addin(bundle):
    assert bundle.testzip() is None
    names = set(bundle.namelist())
    for expected in (
        "INSTALL.txt",
        "install_windows.bat",
        "install_macos.command",
        "AddIns/pic_to_bin/pic_to_bin.py",
        "AddIns/pic_to_bin/_bin_builder.py",
        "AddIns/pic_to_bin/pic_to_bin.manifest",
        "AddIns/pic_to_bin/resources/pic_to_bin/64x64.png",
    ):
        assert expected in names, f"{expected} missing from the bundle"

    # The add-in is the only Fusion entry point — no stale Scripts/ tree.
    stray = [n for n in names if "/" in n and not n.startswith("AddIns/")]
    assert not stray, f"unexpected top-level trees: {stray}"


def test_installer_scripts_keep_their_permissions_and_line_endings(bundle):
    mac = bundle.getinfo("install_macos.command")
    # 0o755, or Finder refuses to launch it on double-click.
    assert (mac.external_attr >> 16) & 0o777 == 0o755
    assert b"\r\n" not in bundle.read("install_macos.command")

    bat = bundle.getinfo("install_windows.bat")
    assert b"\r\n" in bundle.read("install_windows.bat")

    # Add-in files land readable, not the 0600 that writestr() defaults to.
    addin = bundle.getinfo("AddIns/pic_to_bin/pic_to_bin.py")
    assert (addin.external_attr >> 16) & 0o777 == 0o644


def test_addin_source_dir_ships_with_the_package():
    """A wheel that drops these files turns the endpoint into a 500."""
    assert (server._FUSION_ADDIN_DIR / "pic_to_bin.py").is_file()
    assert (server._FUSION_ADDIN_DIR / "_bin_builder.py").is_file()
    assert (server._INSTALLERS_DIR / "install_macos.command").is_file()
