"""Pins for guided login: which browser it opens, where its profile lives, and the launch."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.platforms.browser import cdp, cookies as bc, login
from pipeline.ingestion.utils import SyncAuthError


@pytest.fixture(autouse=True)
def isolated_opyt_home(tmp_path, monkeypatch):
    home = tmp_path / "opyt_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)
    return home


@pytest.fixture
def apps(tmp_path, monkeypatch):
    """Install fake .app executables by bundle name; returns {app: exe}."""
    root = tmp_path / "Applications"
    monkeypatch.setattr(bc, "_APP_DIRS", (root,))

    def install(*names):
        out = {}
        for name in names:
            exe = root / f"{name}.app" / "Contents" / "MacOS" / name
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text("")
            out[name] = exe
        return out

    return install


def _opyt(key):
    b = bc.backend_for(key)
    return replace(b, key=b.key + bc.OPYT_KEY_SUFFIX, label=f"Opyt login — {b.label}")


def test_target_backend_is_the_first_launchable_real_browser(monkeypatch, apps):
    apps("Google Chrome", "Brave Browser")
    monkeypatch.setattr(bc, "installed_backends", lambda: [
        _opyt("chrome"), bc.backend_for("edge"), bc.backend_for("brave"),
        bc.backend_for("chrome")])
    assert login.target_backend() == bc.backend_for("brave")


def test_target_backend_skips_opyt_profiles_and_unlaunchable_browsers(monkeypatch, apps):
    apps("Google Chrome")
    monkeypatch.setattr(bc, "installed_backends", lambda: [
        _opyt("chrome"), bc.backend_for("edge")])
    assert login.target_backend() is None


@pytest.mark.usefixtures("apps")
def test_target_backend_none_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(bc, "installed_backends", list)
    assert login.target_backend() is None


def test_session_dir_is_one_directory_per_browser_key_under_the_home(isolated_opyt_home):
    assert login.session_dir(bc.backend_for("chrome")) == (
        isolated_opyt_home / "browser-sessions" / "chrome")
    assert login.session_dir(bc.backend_for("brave")) == (
        bc.opyt_session_root() / "brave")


def test_start_opens_the_target_browser_on_its_session_dir(monkeypatch, apps,
                                                          isolated_opyt_home):
    exes = apps("Brave Browser")
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("brave")])
    calls = []
    monkeypatch.setattr(cdp, "launch", lambda app_path, user_data_dir, *, url=None: (
        calls.append((app_path, user_data_dir, url)) or object()))

    got = login.start("https://x.com/i/flow/login")

    assert got == bc.backend_for("brave")
    assert calls == [(exes["Brave Browser"], isolated_opyt_home / "browser-sessions" / "brave",
                      "https://x.com/i/flow/login")]


@pytest.mark.usefixtures("apps")
def test_start_raises_sync_auth_error_and_launches_nothing_without_a_browser(monkeypatch):
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("edge")])
    calls = []
    monkeypatch.setattr(cdp, "launch", lambda *a, **_k: calls.append(a))

    with pytest.raises(SyncAuthError) as ei:
        login.start("https://x.com/i/flow/login")

    assert str(ei.value) == (
        "Opyt could not find launchable Chromium to create its managed login profile — it "
        "needs Chrome, Brave, Edge, Vivaldi or Opera installed. Logging into an existing "
        "browser cannot connect a source to OPYT; install one, then call onboard(source=...) "
        "again.")
    assert calls == []


def test_start_returns_without_waiting_on_the_launched_process(monkeypatch, apps):
    apps("Google Chrome")
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("chrome")])

    class _NeverWait:
        def wait(self, *_a, **_k):
            pytest.fail("start waited on the window")

        def terminate(self):
            pytest.fail("start closed the window")

    monkeypatch.setattr(cdp, "launch", lambda *_a, **_k: _NeverWait())
    assert login.start("https://substack.com/sign-in") == bc.backend_for("chrome")


# ── choose_launch_target: one row per case ───────────────────────────────────────

def test_choose_launch_target_takes_the_first_launchable_in_priority_order():
    chrome, brave = bc.backend_for("chrome"), bc.backend_for("brave")
    assert login.choose_launch_target([(chrome, False), (brave, True)]) is brave


def test_choose_launch_target_never_picks_an_opyt_profile():
    managed = replace(bc.backend_for("chrome"), key="chrome" + bc.OPYT_KEY_SUFFIX)
    assert login.choose_launch_target([(managed, True)]) is None


def test_choose_launch_target_none_when_nothing_is_launchable():
    assert login.choose_launch_target([(bc.backend_for("chrome"), False)]) is None
    assert login.choose_launch_target([]) is None
