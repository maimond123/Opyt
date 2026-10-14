"""
tests/test_install_client.py — uninstalling OPYT from a client.

The one property pinned here: a removed OPYT leaves nothing relaunching.
The worker is put back whenever it is missing, so stopping using OPYT is the only
way to have no background process — and that has to actually remove it. But only once no
client still uses OPYT: removing it from one app must not stop the worker another depends on.
"""
from __future__ import annotations

import json

import pytest

from opyt_core import install_client, install_worker


@pytest.fixture
def clients(monkeypatch, tmp_path):
    """Every known client's config redirected into a tmp dir, on macOS."""
    paths = {name: tmp_path / name / "mcp.json" for name in install_client.CLIENTS}
    monkeypatch.setattr(install_client, "CLIENTS", paths)
    monkeypatch.setattr("sys.platform", "darwin")
    return paths


def _register(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {install_client.SERVER_KEY: {"command": "uvx"}}}))


def _worker_removals(monkeypatch) -> list:
    removed = []
    monkeypatch.setattr(install_worker, "uninstall",
                        lambda **kw: removed.append(kw) or {"status": "REMOVED", "path": "p"})
    return removed


def test_removing_opyt_from_its_last_client_removes_the_worker(clients, monkeypatch):
    removed = _worker_removals(monkeypatch)
    _register(clients["cursor"])

    assert install_client.main(["--cursor", "--uninstall"]) == 0

    assert removed, "a removed OPYT must leave nothing relaunching at login"


def test_the_worker_stays_while_another_client_still_uses_opyt(clients, monkeypatch):
    removed = _worker_removals(monkeypatch)
    _register(clients["cursor"])
    _register(clients["claude-desktop"])

    install_client.main(["--cursor", "--uninstall"])

    assert removed == []
    assert install_client.remove_worker_if_unused()["still_registered"] == ["claude-desktop"]


def test_an_unreadable_config_keeps_the_worker(clients, monkeypatch):
    """"Still in use" cannot be ruled out, and removing the worker under a client that still
    needs it stalls everything it was asked to do."""
    removed = _worker_removals(monkeypatch)
    clients["windsurf"].parent.mkdir(parents=True)
    clients["windsurf"].write_text("{ not json")

    assert install_client.remove_worker_if_unused()["status"] == "KEPT"
    assert removed == []


def test_installing_never_touches_the_worker(clients, monkeypatch):
    removed = _worker_removals(monkeypatch)
    monkeypatch.setattr(install_client, "uvx_command", lambda ep: ["/bin/uvx", "--from", "x", ep])

    install_client.main(["--cursor"])

    assert removed == []


def test_off_macos_there_is_no_worker_to_remove(clients, monkeypatch):
    removed = _worker_removals(monkeypatch)
    monkeypatch.setattr("sys.platform", "linux")

    assert install_client.remove_worker_if_unused()["status"] == "NOT_APPLICABLE"
    assert removed == []
