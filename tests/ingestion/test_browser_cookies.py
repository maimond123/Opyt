"""
tests/ingestion/test_browser_cookies.py

Offline tests for the managed-session cookie reader. No live browser: we stub
`installed_backends`, the Chromium row-presence probe and `_read_one`, so the managed
selection (one, several, none) and the failure classification (launch vs other vs
not-connected → actionable remediation) are exercised deterministically.

Two things this module exists to pin:
  1. the fix for "can't tell not-logged-in from permission denied" — the failure-kind →
     remediation path;
  2. the split between DETECTION and READ. Detection is a row-presence check that never
     decrypts (that is what removed the macOS Keychain dialog); exactly one profile — the
     chosen one — is ever decrypted, by launching the browser itself.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.platforms.browser import cdp, cookies as bc
from pipeline.ingestion.utils import SyncAuthError


@pytest.fixture(autouse=True)
def isolated_opyt_home(tmp_path, monkeypatch):
    """Sandbox $OPYT_HOME so the dev machine's real config can't leak in, and clear the
    retired browser override env. It also keeps `opyt_session_backends()` empty, so no real
    guided-login profile leaks in."""
    home = tmp_path / "opyt_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)
    monkeypatch.delenv("OPYT_COOKIES_X", raising=False)
    monkeypatch.delenv("OPYT_COOKIES_SUBSTACK", raising=False)


def _backend(key):
    return bc.backend_for(key)


def _stub_reads(monkeypatch, *, installed, reads):
    """Wire installed_backends(), the Chromium presence probe, and _read_one().

    `installed` = list of backend keys present. `reads` maps (browser, profile) →
    either a cookies dict (success) or an Exception (blocked). A (browser, profile)
    absent from `reads` reads as logged-out. Each backend is given one fake profile
    "Default".

    Chromium cookie paths are synthetic (`/stub/<browser>/Default/Cookies`) so nothing
    can touch this machine's real cookie stores."""
    backs = [bc.backend_for(k) for k in installed]
    by_base = {b.base: b.key for b in backs}
    monkeypatch.setattr(bc, "installed_backends", lambda: backs)
    monkeypatch.setattr(bc, "_chromium_cookie_files", lambda base: [
        ("Default", Path(f"/stub/{by_base[base]}/Default/Cookies"))])

    def fake_present(cookie_file, domains, name):
        val = reads.get((Path(cookie_file).parts[2], "Default"))
        if isinstance(val, Exception):
            raise val
        return isinstance(val, dict) and bool(val.get(name))

    def fake_read_one(backend, domains, *, cookie_file=None):
        val = reads.get((backend.key, "Default"))
        if isinstance(val, Exception):
            return {}, val
        if isinstance(val, dict):
            return dict(val), None
        return {}, None

    monkeypatch.setattr(bc, "_chromium_has_cookie", fake_present)
    monkeypatch.setattr(bc, "_read_one", fake_read_one)


# ── build_cookie_header unchanged ────────────────────────────────────────────────

def test_build_cookie_header_unchanged():
    assert bc.build_cookie_header({"a": "1", "b": "2"}) == "a=1; b=2"
    assert bc.build_cookie_header({}) == ""


# ── Detection never decrypts; the chosen profile is read exactly once ────────────

def test_detection_does_not_decrypt(monkeypatch):
    """⚠️ THE POINT OF THE CHANGE. Scanning must not call the decrypting read for a
    Chromium profile — that read launches a browser, and the Python-side decryption it
    replaced is what raised the macOS Keychain dialog."""
    _stub_reads(monkeypatch, installed=["chrome"],
                reads={("chrome", "Default"): {"sessionKey": "tok"}})
    reads = []
    inner = bc._read_one
    monkeypatch.setattr(bc, "_read_one",
                        lambda b, d, *, cookie_file=None: reads.append(b.key) or inner(
                            b, d, cookie_file=cookie_file))
    candidates, failures = bc.list_logged_in(["claude.ai"], "sessionKey")
    assert [c["profile"] for c in candidates] == ["Default"]
    assert reads == []                      # nothing decrypted just to detect
    assert "cookies" not in candidates[0]   # discovery hands back no session secrets


# ── The exact-domain rule (both forms) ───────────────────────────────────────────

def test_domain_match_rejects_a_bare_suffix():
    """⚠️ MEASURED BUG. `LIKE '%x.com'` also matches adgrx.com — the spike's first run
    transplanted 22 unrelated ad-tracker rows that way."""
    assert bc._domain_matches("x.com", ["x.com"])
    assert bc._domain_matches(".x.com", ["x.com"])
    assert bc._domain_matches("api.x.com", ["x.com"])
    assert not bc._domain_matches("adgrx.com", ["x.com"])
    assert not bc._domain_matches("notx.com", ["x.com"])


def test_domain_sql_is_the_same_rule(tmp_path):
    """The SQL twin must agree with the Python one — they are one rule in two forms."""
    db = tmp_path / "Cookies"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, "
                 "encrypted_value BLOB)")
    hosts = ["x.com", ".x.com", "api.x.com", "adgrx.com", "notx.com", "twitter.com"]
    conn.executemany("INSERT INTO cookies VALUES (?, 'auth_token', '', X'0102')",
                     [(h,) for h in hosts])
    conn.commit()
    where, params = bc._domain_sql(["x.com"])
    got = {r[0] for r in conn.execute(f"SELECT host_key FROM cookies WHERE {where}", params)}
    conn.close()
    assert got == {h for h in hosts if bc._domain_matches(h, ["x.com"])}
    assert got == {"x.com", ".x.com", "api.x.com"}


def test_prune_deletes_everything_outside_the_domains(tmp_path):
    """The temp profile the browser is launched against must hold ONLY the session asked
    for — it is a decrypted copy of the user's cookies, so its contents are the blast
    radius if anything ever escapes the temp dir."""
    db = tmp_path / "Cookies"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT)")
    conn.executemany("INSERT INTO cookies VALUES (?, 'c')",
                     [("x.com",), ("adgrx.com",), ("mail.google.com",), (".twitter.com",)])
    conn.commit()
    conn.close()
    bc._prune_to_domains(db, ["x.com", "twitter.com"])
    conn = sqlite3.connect(db)
    left = {r[0] for r in conn.execute("SELECT host_key FROM cookies")}
    conn.close()
    assert left == {"x.com", ".twitter.com"}


# ── Presence probe: row exists, value stays encrypted ────────────────────────────

def test_presence_probe_reads_rows_not_values(tmp_path):
    """The probe must see an encrypted cookie (value empty, encrypted_value set) as a
    session. That is exactly the row a decrypt would have been needed to read."""
    db = tmp_path / "Cookies"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, "
                 "encrypted_value BLOB)")
    conn.execute("INSERT INTO cookies VALUES ('.x.com', 'auth_token', '', X'763130ab')")
    conn.execute("INSERT INTO cookies VALUES ('.x.com', 'empty_one', '', X'')")
    conn.commit()
    conn.close()
    assert bc._chromium_has_cookie(db, ["x.com"], "auth_token") is True
    assert bc._chromium_has_cookie(db, ["x.com"], "empty_one") is False
    assert bc._chromium_has_cookie(db, ["x.com"], "sessionKey") is False
    assert bc._chromium_has_cookie(db, ["adgrx.com"], "auth_token") is False


# ── OPYT's own login profiles (created by guided login) ──────────────────────────

def _managed_session(backend: str) -> None:
    profile = bc.opyt_session_root() / backend / "Default"
    profile.mkdir(parents=True)
    (profile / "Cookies").touch()


def test_read_opyt_cookies_uses_only_an_opyt_session(monkeypatch, tmp_path):
    """A normal Chrome login must never substitute for OPYT's dedicated X session."""
    normal = replace(bc.backend_for("chrome"), base=tmp_path / "normal-chrome")
    normal_profile = normal.base / "Default"
    normal_profile.mkdir(parents=True)
    (normal_profile / "Cookies").touch()
    _managed_session("chrome")
    managed = bc.opyt_session_backends()
    monkeypatch.setattr(bc, "installed_backends",
                        lambda: [normal, *managed])
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *args: True)
    monkeypatch.setattr(
        bc, "_read_one",
        lambda backend, domains, *, cookie_file=None: (
            {"auth_token": "managed" if backend.key.endswith("@opyt") else "normal"}, None
        ),
    )

    assert bc.read_opyt_cookies(["x.com", "twitter.com"], "auth_token", source="X") == {
        "auth_token": "managed"
    }


def test_read_opyt_cookies_tells_the_user_to_connect_when_absent(monkeypatch):
    monkeypatch.setattr(bc, "installed_backends", lambda: [])

    with pytest.raises(SyncAuthError, match=r"onboard\(source="):
        bc.read_opyt_cookies(["x.com", "twitter.com"], "auth_token", source="X")


def test_read_opyt_cookies_rejects_multiple_managed_sessions(monkeypatch):
    _managed_session("chrome")
    _managed_session("brave")
    managed = bc.opyt_session_backends()
    monkeypatch.setattr(bc, "installed_backends", lambda: managed)
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *args: True)

    with pytest.raises(SyncAuthError, match="Multiple OPYT-managed X sessions"):
        bc.read_opyt_cookies(["x.com", "twitter.com"], "auth_token", source="X")

def test_opyt_session_profiles_are_read_like_any_other(monkeypatch, tmp_path):
    """A guided login leaves a normal Chromium user-data-dir, so it is picked up by the
    same enumeration — one read path, and one answer to 'is there a session'."""
    root = bc.opyt_session_root() / "chrome" / "Default"
    root.mkdir(parents=True)
    (root / "Cookies").touch()
    backs = bc.opyt_session_backends()
    assert [b.key for b in backs] == ["chrome@opyt"]
    assert backs[0].base == bc.opyt_session_root() / "chrome"
    assert bc.backend_for("chrome@opyt") == backs[0]
    assert [p for p, _ in bc._chromium_cookie_files(backs[0].base)] == ["Default"]


def test_a_real_browser_session_outranks_the_opyt_one(monkeypatch, tmp_path):
    """Ordering matters: a guided login must never quietly win over (or overwrite) a
    session in the user's own browser."""
    (bc.opyt_session_root() / "chrome").mkdir(parents=True)
    keys = [b.key for b in bc.installed_backends()]
    assert "chrome@opyt" in keys
    assert keys[-1] == "chrome@opyt"


def test_a_browser_opyt_cannot_launch_is_not_offered(monkeypatch):
    """A Chromium profile root with no app to launch is a backend that cannot work —
    dropping it beats surfacing a candidate whose read is guaranteed to fail."""
    monkeypatch.setattr(bc.BrowserBackend, "app_path", lambda self: None)
    assert [b.key for b in bc.installed_backends() if b.base is not None] == []


# ── Managed-session failure paths ────────────────────────────────────────────────
# Pinned on `read_opyt_cookies` so the failure copy survives the retirement of the generic
# reader, whose tests above are the only other place these paths run.

def _managed_only(monkeypatch, *backends: str) -> None:
    for b in backends:
        _managed_session(b)
    monkeypatch.setattr(bc, "installed_backends", lambda: bc.opyt_session_backends())


def test_read_opyt_cookies_reports_a_launch_failure_as_one(monkeypatch):
    _managed_only(monkeypatch, "chrome")
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *_: True)
    monkeypatch.setattr(bc, "_read_one", lambda _b, _d, **_: (
        {}, cdp.CDPError("browser exited (code 1) before opening DevTools")))
    with pytest.raises(SyncAuthError) as ei:
        bc.read_opyt_cookies(["x.com"], "auth_token", source="X")
    msg = str(ei.value)
    assert "could not start Opyt login — Chrome" in msg and "/Applications" in msg


def test_read_opyt_cookies_read_back_empty_is_not_logged_out(monkeypatch):
    _managed_only(monkeypatch, "chrome")
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *_: True)
    monkeypatch.setattr(bc, "_read_one", lambda _b, _d, **_: ({}, None))
    with pytest.raises(SyncAuthError) as ei:
        bc.read_opyt_cookies(["x.com"], "auth_token", source="X")
    assert "chrome@opyt/Default" in str(ei.value) and "returned nothing" in str(ei.value)


def test_read_opyt_cookies_reads_only_the_managed_cookie_file(monkeypatch):
    _managed_only(monkeypatch, "chrome")
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda *_: True)
    reads = []
    monkeypatch.setattr(bc, "_read_one", lambda _b, _d, *, cookie_file=None: (
        reads.append(cookie_file) or ({"auth_token": "t"}, None)))
    assert bc.read_opyt_cookies("x.com", "auth_token", source="X") == {"auth_token": "t"}
    assert reads == [bc.opyt_session_root() / "chrome" / "Default" / "Cookies"]


def test_read_opyt_cookies_blocked_probe_gives_its_remediation(monkeypatch):
    """A probe that raised is a failure, not a logged-out session: the user is told which
    session could not be read, then sent to onboard, the only way to repair a managed one."""
    _managed_only(monkeypatch, "chrome")

    def blocked(*_):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(bc, "_chromium_has_cookie", blocked)
    with pytest.raises(SyncAuthError) as ei:
        bc.read_opyt_cookies(["x.com"], "auth_token", source="X")
    msg = str(ei.value)
    assert msg.startswith("Couldn't read your Opyt login — Chrome session for X.")
    assert "onboard(source='x')" in msg


def test_read_opyt_cookies_surfaces_the_most_actionable_failure(monkeypatch):
    """Two blocked managed profiles: the launch failure outranks the unclassified one."""
    _managed_only(monkeypatch, "brave", "chrome")

    def probe(cookie_file, _domains, _name):
        if Path(cookie_file).parts[-3] == "chrome":
            raise cdp.CDPError("browser exited")
        raise RuntimeError("weird")

    monkeypatch.setattr(bc, "_chromium_has_cookie", probe)
    with pytest.raises(SyncAuthError) as ei:
        bc.read_opyt_cookies(["x.com"], "auth_token", source="X")
    assert "could not start Opyt login — Chrome" in str(ei.value)


def test_a_chromium_read_failure_is_classified_by_type():
    assert bc._classify_failure(cdp.CDPError("boom")) == "browser_launch"
    assert bc._classify_failure(Exception("keychain error")) == "other"
    assert bc._classify_failure(PermissionError("x")) == "other"


# ── choose_managed_candidate: one row per case ───────────────────────────────────

def _candidate(browser):
    return {"browser": browser, "profile": "Default", "label": browser,
            "backend": _backend("chrome"), "cookie_file": Path("/stub/Cookies")}


def test_choose_managed_candidate_uses_the_one_candidate():
    one = _candidate("chrome@opyt")
    assert bc.choose_managed_candidate([one], [], "X") is one


def test_choose_managed_candidate_refuses_several():
    with pytest.raises(SyncAuthError, match="Multiple OPYT-managed X sessions"):
        bc.choose_managed_candidate([_candidate("chrome@opyt"), _candidate("brave@opyt")], [], "X")


def test_choose_managed_candidate_none_with_failures_gives_the_worst_remediation():
    failures = [{"browser": "brave", "profile": "Default", "backend": _backend("brave"),
                 "kind": "other", "detail": ""},
                {"browser": "chrome", "profile": "Default", "backend": _backend("chrome"),
                 "kind": "browser_launch", "detail": ""}]
    with pytest.raises(SyncAuthError, match="could not start Chrome"):
        bc.choose_managed_candidate([], failures, "X")


def test_choose_managed_candidate_none_without_failures_says_connect():
    with pytest.raises(SyncAuthError, match=r"onboard\(source='x'\)"):
        bc.choose_managed_candidate([], [], "X")


def test_remediation_for_an_unclassified_failure_points_at_onboard_not_a_browser_list():
    """Reads are managed-only: logging into a browser of the user's own fixes nothing."""
    msg = bc.remediation("other", None, "X")
    assert "onboard(source='x')" in msg
    assert "Chrome/Brave" not in msg and "supported browser" not in msg


# ── A session handed in through the environment ──────────────────────────────────────────────

def _no_profile_scan(monkeypatch):
    def refuse():
        raise AssertionError("a profile was scanned although the environment holds the session")
    monkeypatch.setattr(bc, "installed_backends", refuse)
    monkeypatch.setattr(bc, "opyt_session_backends", refuse)


def test_an_environment_session_is_read_without_scanning_a_profile(monkeypatch):
    monkeypatch.setenv("OPYT_COOKIES_X", " auth_token=aaa ; ct0=bbb ")
    _no_profile_scan(monkeypatch)
    assert bc.read_opyt_cookies(["x.com", "twitter.com"], "auth_token", source="X") == {
        "auth_token": "aaa", "ct0": "bbb"}
    candidates, failures = bc.list_opyt_logged_in(["x.com", "twitter.com"], "auth_token")
    assert len(candidates) == 1 and failures == []


def test_the_site_name_picks_the_variable(monkeypatch):
    monkeypatch.setenv("OPYT_COOKIES_SUBSTACK", "substack.sid=s1")
    _no_profile_scan(monkeypatch)
    assert bc.read_opyt_cookies("substack.com", "substack.sid", source="substack") == {
        "substack.sid": "s1"}


def test_an_environment_session_without_its_auth_cookie_is_refused(monkeypatch):
    """A half-pasted session is not connected, and reading it says why instead of sending a
    Cookie header that 401s later."""
    monkeypatch.setenv("OPYT_COOKIES_X", "ct0=bbb")
    _no_profile_scan(monkeypatch)
    assert bc.list_opyt_logged_in(["x.com"], "auth_token") == ([], [])
    with pytest.raises(SyncAuthError, match="no auth_token cookie"):
        bc.read_opyt_cookies(["x.com"], "auth_token", source="X")
