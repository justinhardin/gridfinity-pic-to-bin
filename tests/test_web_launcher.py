"""Tests for the ``pic-to-bin-web`` console-script shim.

The web dependencies are core, so the shim's guard is for installs that
lack them anyway — pruned venvs, or upgrades from the releases that kept
them in a ``[web]`` extra. It has to turn that into repair instructions
without hiding real import bugs.
"""

from __future__ import annotations

import sys

import pytest

from pic_to_bin.web import launcher


class _Blocker:
    """meta_path finder that fails one module the way a missing dep does."""

    def __init__(self, blocked: str):
        self.blocked = blocked

    # Argument order matches how the import system calls finders.
    def find_spec(self, name, path=None, target=None):
        if name == self.blocked or name.startswith(self.blocked + "."):
            raise ImportError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture
def block(monkeypatch):
    """Make importing ``target`` fail, even if it is already imported.

    Both evictions matter: server.py must re-run its imports, and the
    blocked module must leave sys.modules or the import short-circuits
    before any meta_path finder is consulted.
    """
    def _apply(target: str):
        monkeypatch.delitem(sys.modules, "pic_to_bin.web.server", raising=False)
        monkeypatch.delitem(sys.modules, target, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_Blocker(target), *sys.meta_path])
    return _apply


def test_missing_web_dependency_explains_the_repair(block):
    block("fastapi")
    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    msg = str(excinfo.value)
    assert "missing: fastapi" in msg
    assert "pipx install --force gridfinity-pic-to-bin" in msg
    # No extra to suggest any more — the deps ship with the package.
    assert "[web]" not in msg.replace("a separate [web] extra", "")


def test_a_real_import_bug_still_raises(block):
    """Don't blame the extra for a broken import inside our own code."""
    block("pic_to_bin.web.server")
    with pytest.raises(ImportError):
        launcher.main()


def test_main_delegates_to_server_cli(monkeypatch):
    import pic_to_bin.web.server as server

    called = []
    monkeypatch.setattr(server, "cli", lambda: called.append(True))
    launcher.main()
    assert called == [True]


def test_enable_llm_without_the_sdk_exits_with_instructions(monkeypatch, tmp_path):
    """--enable-llm is checked at startup, not at the first request.

    anthropic is the one dependency that stayed optional, so passing the
    flag without it has to say so before the server starts serving.
    """
    import importlib.util

    from pic_to_bin.web import server

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "anthropic" else real_find_spec(name, *a, **kw),
    )
    monkeypatch.setattr(sys, "argv", ["pic-to-bin-web", "--enable-llm"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        server.cli()

    msg = str(excinfo.value)
    assert "anthropic" in msg
    assert '"gridfinity-pic-to-bin[llm]"' in msg
