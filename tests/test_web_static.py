"""Tests for the frontend's script loading and the CSP that governs it.

The failure these guard against is silent and total: if the import map names
an origin the Content-Security-Policy doesn't allow, the browser blocks Lit
and the app renders a blank page — the server logs nothing and every
server-side test still passes.
"""

from __future__ import annotations

import pytest

from pic_to_bin.web import server

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client(tmp_path):
    with TestClient(server.create_app(tmp_path / "jobs")) as c:
        yield c


def _script_src(response) -> str:
    csp = response.headers["content-security-policy"]
    return csp.split("script-src ")[1].split(";")[0]


def test_lit_bundle_is_vendored_and_self_contained():
    """The shipped bundle must not phone home.

    jsdelivr's ``/npm/lit/+esm`` looks like a vendored copy but is a ~500 byte
    stub that re-imports the real modules from the CDN at runtime — which the
    CSP then blocks. Assert on content, not just on the file existing.
    """
    assert server.LIT_BUNDLE.is_file()
    text = server.LIT_BUNDLE.read_text(encoding="utf-8")
    assert "export{" in text
    assert "LitElement" in text
    assert "//" + "# sourceMappingURL" not in text
    for host in ("cdn.jsdelivr.net", "esm.sh", "unpkg.com", "//npm/"):
        assert host not in text, f"bundle still references {host}"


def test_app_page_import_map_matches_csp(client):
    r = client.get("/app")
    assert r.status_code == 200
    assert '"lit": "/static/lit-all.min.js"' in r.text
    # Same-origin only: no CDN in the trust path when Lit is vendored.
    assert "esm.sh" not in _script_src(r)
    assert client.get("/static/lit-all.min.js").status_code == 200


def test_csp_widens_when_the_bundle_is_missing(client, monkeypatch):
    """Fallback path: import map and CSP have to move together."""
    monkeypatch.setattr(server, "_lit_is_vendored", lambda: False)
    r = client.get("/app")
    assert r.status_code == 200
    assert server.LIT_CDN_URL in r.text
    assert server.LIT_CDN_ORIGIN in _script_src(r)
    assert r.headers["content-length"] == str(len(r.content))


def test_devtools_workspace_endpoint_only_answers_from_a_checkout(
    client, monkeypatch, tmp_path
):
    """404 for installed copies — site-packages is not a project root, and
    DevTools reports it as ``Unable to add filesystem: <illegal path>``."""
    url = "/.well-known/appspecific/com.chrome.devtools.json"
    assert client.get(url).status_code == 200

    # Stand in for a pipx/pip install: no pyproject.toml above the package.
    installed = tmp_path / "site-packages" / "pic_to_bin" / "web" / "static"
    installed.mkdir(parents=True)
    monkeypatch.setattr(server, "STATIC_DIR", installed)
    assert client.get(url).status_code == 404
