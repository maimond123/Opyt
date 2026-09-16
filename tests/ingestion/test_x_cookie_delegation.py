"""X cookie access is limited to OPYT's persistent browser session."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.ingestion import browser_cookies as bc
from pipeline.ingestion import x_graphql as bookmarks
from pipeline.ingestion import x_graphql_core as core


def clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "opyt_home"))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)


def test_has_managed_x_session_ignores_normal_browser_cookie(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    normal = replace(bc.backend_for("chrome"), base=tmp_path / "normal-chrome")
    normal_profile = normal.base / "Default"
    normal_profile.mkdir(parents=True)
    (normal_profile / "Cookies").touch()
    monkeypatch.setattr(bc, "installed_backends",
                        lambda: [normal, *bc.opyt_session_backends()])
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *args: True)

    assert bookmarks.has_managed_x_session() is False

    managed_profile = bc.opyt_session_root() / "chrome" / "Default"
    managed_profile.mkdir(parents=True)
    (managed_profile / "Cookies").touch()

    assert bookmarks.has_managed_x_session() is True


def test_x_cookie_reader_uses_the_managed_reader(monkeypatch):
    monkeypatch.setattr(bc, "read_cookies",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("X attempted to read a normal browser session")))
    monkeypatch.setattr(bc, "read_opyt_cookies",
                        lambda domains, auth_cookie, *, source: {
                            "auth_token": "managed", "ct0": "csrf"})

    assert core.read_x_cookies() == {"auth_token": "managed", "ct0": "csrf"}
