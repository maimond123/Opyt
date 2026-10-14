"""Pins for the Chromium transplant read (`_read_chromium`) and `_read_one`'s Chromium branch.

The browser launch is stubbed at `cdp.controlled_browser` / `cdp.get_cookies`; the cookie DB
copy, prune and temp-profile cleanup run for real against a SQLite file under tmp_path.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.platforms.browser import cdp, cookies as bc


@pytest.fixture(autouse=True)
def isolated_opyt_home(tmp_path, monkeypatch):
    home = tmp_path / "opyt_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)


@pytest.fixture
def apps(tmp_path, monkeypatch):
    """A fake /Applications holding Google Chrome; returns its executable."""
    root = tmp_path / "Applications"
    exe = root / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(bc, "_APP_DIRS", (root,))
    return exe


def _cookie_db(path: Path, hosts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, "
                 "encrypted_value BLOB)")
    conn.executemany("INSERT INTO cookies VALUES (?, 'c', '', X'01')", [(h,) for h in hosts])
    conn.commit()
    conn.close()
    return path


def _hosts(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {r[0] for r in conn.execute("SELECT host_key FROM cookies")}
    finally:
        conn.close()


def _stub_browser(monkeypatch, cookies, *, seen: dict, raises: Exception | None = None):
    @contextmanager
    def _controlled(exe, *, user_data_dir=None, headless=True, url=None):
        seen["exe"] = exe
        seen["data_dir"] = user_data_dir
        seen["headless"] = headless
        seen["url"] = url
        seen["files"] = sorted(str(p.relative_to(user_data_dir))
                               for p in user_data_dir.rglob("*") if p.is_file())
        for rel in ("Default/Network/Cookies", "Default/Cookies"):
            if (user_data_dir / rel).exists():
                seen["hosts"] = _hosts(user_data_dir / rel)
        if raises is not None:
            raise raises
        yield "session"

    def _get_cookies(session):
        seen["session"] = session
        return cookies

    monkeypatch.setattr(cdp, "controlled_browser", _controlled)
    monkeypatch.setattr(cdp, "get_cookies", _get_cookies)


# ── _read_chromium ───────────────────────────────────────────────────────────────

def test_read_chromium_raises_when_the_browser_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "_APP_DIRS", (tmp_path / "empty",))
    with pytest.raises(cdp.CDPError) as ei:
        bc._read_chromium(bc.backend_for("chrome"), ["x.com"], tmp_path / "Cookies")
    assert str(ei.value) == "Chrome is not installed where Opyt can launch it"


def test_read_chromium_transplants_a_pruned_copy_and_filters_the_reply(monkeypatch, tmp_path,
                                                                      apps):
    src = _cookie_db(tmp_path / "profile" / "Default" / "Network" / "Cookies",
                     ["x.com", ".x.com", "api.x.com", "adgrx.com", "google.com"])
    seen: dict = {}
    _stub_browser(monkeypatch, [
        {"name": "auth_token", "value": "tok", "domain": ".x.com"},
        {"name": "ct0", "value": "first", "domain": "x.com"},
        {"name": "ct0", "value": "second", "domain": "api.x.com"},
        {"name": "ad", "value": "no", "domain": "adgrx.com"},
        {"name": "nodomain", "value": "no"},
    ], seen=seen)

    got = bc._read_chromium(bc.backend_for("chrome"), ["x.com"], src)

    assert got == {"auth_token": "tok", "ct0": "second"}
    assert seen["exe"] == apps
    assert seen["headless"] is True and seen["url"] is None
    assert seen["session"] == "session"
    assert seen["data_dir"].name.startswith("opyt-transplant-")
    assert seen["files"] == ["Default/Network/Cookies"]
    assert seen["hosts"] == {"x.com", ".x.com", "api.x.com"}
    assert not seen["data_dir"].exists()
    assert _hosts(src) == {"x.com", ".x.com", "api.x.com", "adgrx.com", "google.com"}


@pytest.mark.usefixtures("apps")
def test_read_chromium_mirrors_the_older_layout_and_leaves_no_sidecars(monkeypatch, tmp_path):
    src = _cookie_db(tmp_path / "profile" / "Default" / "Cookies", ["substack.com"])
    src.with_name("Cookies-journal").write_bytes(b"")
    seen: dict = {}
    _stub_browser(monkeypatch, [{"name": "substack.sid", "value": "s",
                                 "domain": ".substack.com"}], seen=seen)

    got = bc._read_chromium(bc.backend_for("chrome"), ["substack.com"], src)

    assert got == {"substack.sid": "s"}
    assert seen["files"] == ["Default/Cookies"]
    assert seen["hosts"] == {"substack.com"}


@pytest.mark.usefixtures("apps")
def test_read_chromium_removes_the_temp_profile_when_the_launch_fails(monkeypatch, tmp_path):
    src = _cookie_db(tmp_path / "profile" / "Default" / "Network" / "Cookies", ["x.com"])
    seen: dict = {}
    boom = cdp.CDPError("browser exited (code 1) before opening DevTools")
    _stub_browser(monkeypatch, [], seen=seen, raises=boom)

    with pytest.raises(cdp.CDPError) as ei:
        bc._read_chromium(bc.backend_for("chrome"), ["x.com"], src)

    assert ei.value is boom
    assert not seen["data_dir"].exists()


def test_read_chromium_launches_the_opyt_profile_backend_by_its_own_app(monkeypatch, tmp_path,
                                                                       apps):
    backend = replace(bc.backend_for("chrome"), key="chrome" + bc.OPYT_KEY_SUFFIX,
                      base=tmp_path / "sessions" / "chrome")
    src = _cookie_db(backend.base / "Default" / "Cookies", ["x.com"])
    seen: dict = {}
    _stub_browser(monkeypatch, [{"name": "auth_token", "value": "t", "domain": "x.com"}],
                  seen=seen)
    assert bc._read_chromium(backend, ["x.com"], src) == {"auth_token": "t"}
    assert seen["exe"] == apps


# ── _read_one, Chromium branch ───────────────────────────────────────────────────

def test_read_one_chromium_returns_the_read_with_no_error(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "_read_chromium", lambda b, d, cf: (
        calls.append((b.key, d, cf)) or {"auth_token": "t"}))
    got = bc._read_one(bc.backend_for("brave"), ["x.com"], cookie_file=Path("/stub/Cookies"))
    assert got == ({"auth_token": "t"}, None)
    assert calls == [("brave", ["x.com"], Path("/stub/Cookies"))]


def test_read_one_chromium_logged_out_is_an_empty_dict_and_no_error(monkeypatch):
    monkeypatch.setattr(bc, "_read_chromium", lambda _b, _d, _cf: {})
    assert bc._read_one(bc.backend_for("chrome"), ["x.com"],
                        cookie_file=Path("/stub/Cookies")) == ({}, None)


@pytest.mark.parametrize("exc", [cdp.CDPError("no DevTools"), sqlite3.DatabaseError("bad"),
                                 FileNotFoundError("gone")])
def test_read_one_chromium_returns_any_failure_as_the_error(monkeypatch, exc):
    def _raise(_b, _d, _cf):
        raise exc

    monkeypatch.setattr(bc, "_read_chromium", _raise)
    cookies, err = bc._read_one(bc.backend_for("chrome"), ["x.com"],
                                cookie_file=Path("/stub/Cookies"))
    assert cookies == {} and err is exc


@pytest.mark.usefixtures("apps")
def test_read_one_chromium_without_a_cookie_file_reports_an_error():
    cookies, err = bc._read_one(bc.backend_for("chrome"), ["x.com"])
    assert cookies == {}
    assert isinstance(err, AttributeError)


@pytest.mark.usefixtures("apps")
def test_read_one_chromium_end_to_end_through_the_transplant(monkeypatch, tmp_path):
    src = _cookie_db(tmp_path / "profile" / "Default" / "Network" / "Cookies", ["x.com"])
    seen: dict = {}
    _stub_browser(monkeypatch, [{"name": "auth_token", "value": "t", "domain": ".x.com"}],
                  seen=seen)
    assert bc._read_one(bc.backend_for("chrome"), ["x.com"], cookie_file=src) == (
        {"auth_token": "t"}, None)
