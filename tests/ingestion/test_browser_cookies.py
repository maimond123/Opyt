"""
tests/ingestion/test_browser_cookies.py

Offline tests for the generalized multi-browser cookie reader. No live browser: we
stub `installed_backends`, the Chromium row-presence probe and `_read_one`, so the
resolution logic (browser/profile override, auto-pick, ambiguity) and the failure
classification (launch vs keychain vs FDA vs not-logged-in → actionable remediation)
are exercised deterministically.

Two things this module exists to pin:
  1. the fix for "can't tell not-logged-in from permission denied" — the failure-kind →
     remediation path;
  2. the split between DETECTION and READ. Detection is a row-presence check that never
     decrypts (that is what removed the macOS Keychain dialog); exactly one profile — the
     chosen one — is ever decrypted, by launching the browser itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pipeline.ingestion import browser_cookies as bc, cdp
from pipeline.ingestion.utils import SyncAuthError


@pytest.fixture(autouse=True)
def isolated_opyt_home(tmp_path, monkeypatch):
    """Sandbox $OPYT_HOME so config resolution (cookie_browser → settings.yaml) can't
    pick up the dev machine's real config, and clear the browser override env. It also
    keeps `opyt_session_backends()` empty, so no real guided-login profile leaks in."""
    home = tmp_path / "opyt_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)
    monkeypatch.delenv("X_CHROME_PROFILE", raising=False)


def _backend(key):
    return bc.backend_for(key)


def _stub_reads(monkeypatch, *, installed, reads):
    """Wire installed_backends(), the Chromium presence probe, and _read_one().

    `installed` = list of backend keys present. `reads` maps (browser, profile) →
    either a cookies dict (success) or an Exception (blocked). A (browser, profile)
    absent from `reads` reads as logged-out. Chromium backends are given one fake
    profile "Default"; delegated backends (Arc/Firefox/Safari) read at profile=None.

    Chromium cookie paths are synthetic (`/stub/<browser>/Default/Cookies`) so nothing
    can touch this machine's real cookie stores."""
    backs = [bc.backend_for(k) for k in installed]
    by_base = {b.base: b.key for b in backs if b.base is not None}
    monkeypatch.setattr(bc, "installed_backends", lambda: backs)
    monkeypatch.setattr(bc, "_chromium_cookie_files", lambda base: [
        ("Default", Path(f"/stub/{by_base[base]}/Default/Cookies"))])
    monkeypatch.setattr(bc, "_chromium_profile_labels", lambda base: {"Default": "Default"})

    def fake_present(cookie_file, domains, name):
        val = reads.get((Path(cookie_file).parts[2], "Default"))
        if isinstance(val, Exception):
            raise val
        return isinstance(val, dict) and bool(val.get(name))

    def fake_read_one(backend, domains, *, cookie_file=None):
        profile = "Default" if backend.base is not None else None
        val = reads.get((backend.key, profile))
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


# ── Auto-pick a lone candidate ───────────────────────────────────────────────────

def test_auto_picks_lone_candidate(monkeypatch):
    _stub_reads(monkeypatch, installed=["chrome", "safari"],
                reads={("chrome", "Default"): {"sessionKey": "tok", "x": "y"}})
    got = bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    assert got == {"sessionKey": "tok", "x": "y"}


def test_no_candidate_not_logged_in(monkeypatch):
    _stub_reads(monkeypatch, installed=["chrome"], reads={})
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    assert "No claude login" in str(ei.value)


# ── Detection never decrypts; the chosen profile is read exactly once ────────────

def test_detection_does_not_decrypt(monkeypatch):
    """⚠️ THE POINT OF THE CHANGE. Scanning must not call the decrypting read for a
    Chromium profile — that read launches a browser, and the browser_cookie3 read it
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


def test_only_the_chosen_profile_is_read(monkeypatch):
    """Eight logged-in Chromium profiles must still cost ONE decrypting read."""
    backs = [bc.backend_for("chrome")]
    monkeypatch.setattr(bc, "installed_backends", lambda: backs)
    monkeypatch.setattr(bc, "_chromium_cookie_files", lambda base: [
        (f"Profile {i}", Path(f"/stub/chrome/Profile {i}/Cookies")) for i in range(8)])
    monkeypatch.setattr(bc, "_chromium_profile_labels", lambda base: {})
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda cf, d, n: True)
    reads = []
    monkeypatch.setattr(bc, "_read_one", lambda b, d, *, cookie_file=None: (
        reads.append(cookie_file) or ({"auth_token": "t"}, None)))
    got = bc.read_cookies(["x.com"], "auth_token", profile="Profile 5", source="X")
    assert got == {"auth_token": "t"}
    assert reads == [Path("/stub/chrome/Profile 5/Cookies")]


def test_read_back_empty_is_not_reported_as_logged_out(monkeypatch):
    """Detection saw the row and the read returned nothing. Returning {} would send an
    empty Cookie header and 401 far from here; "not logged in" would be a lie."""
    _stub_reads(monkeypatch, installed=["chrome"],
                reads={("chrome", "Default"): {"sessionKey": "tok"}})
    monkeypatch.setattr(bc, "_read_one", lambda b, d, *, cookie_file=None: ({}, None))
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    assert "returned nothing" in str(ei.value)


# ── OPYT_BROWSER override wins ────────────────────────────────────────────────────

def test_browser_override_wins(monkeypatch):
    monkeypatch.setenv("OPYT_BROWSER", "brave")
    _stub_reads(monkeypatch, installed=["chrome", "brave"],
                reads={("chrome", "Default"): {"sessionKey": "chrometok"},
                       ("brave", "Default"): {"sessionKey": "bravetok"}})
    got = bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    assert got == {"sessionKey": "bravetok"}


def test_override_scan_skips_other_browsers(monkeypatch):
    """With an override, list_logged_in is told to scan ONLY that browser — so we never
    touch (and never launch) the others."""
    monkeypatch.setenv("OPYT_BROWSER", "chrome")
    scanned = {}

    real_list = bc.list_logged_in

    def spy(domains, auth_cookie, *, browsers=None):
        scanned["browsers"] = browsers
        return real_list(domains, auth_cookie, browsers=browsers)

    _stub_reads(monkeypatch, installed=["chrome", "brave"],
                reads={("chrome", "Default"): {"sessionKey": "t"}})
    monkeypatch.setattr(bc, "list_logged_in", spy)
    bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    assert scanned["browsers"] == ["chrome"]


# ── Priority-first: first browser with a session wins; the rest aren't touched ───

def test_priority_first_prefers_higher_and_stops(monkeypatch):
    """Chrome and Brave both logged in → Chrome (higher priority) wins and Brave is
    never even probed."""
    probed = []
    _stub_reads(monkeypatch, installed=["chrome", "brave"],
                reads={("chrome", "Default"): {"auth_token": "chrome"},
                       ("brave", "Default"): {"auth_token": "brave"}})
    inner = bc._chromium_has_cookie
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda cf, d, n: (
        probed.append(Path(cf).parts[2]) or inner(cf, d, n)))
    got = bc.read_cookies(["x.com"], "auth_token", source="X")
    assert got == {"auth_token": "chrome"}
    assert probed == ["chrome"]


# ── Multiple PROFILES in the winning browser → ambiguity error that LISTS them ───

def test_multiple_profiles_raises_listing(monkeypatch):
    backs = [bc.backend_for("chrome")]
    monkeypatch.setattr(bc, "installed_backends", lambda: backs)
    monkeypatch.setattr(bc, "_chromium_cookie_files", lambda base: [
        ("Default", Path("/stub/chrome/Default/Cookies")),
        ("Profile 1", Path("/stub/chrome/Profile 1/Cookies")),
    ])
    monkeypatch.setattr(bc, "_chromium_profile_labels",
                        lambda base: {"Default": "Personal", "Profile 1": "Work"})
    monkeypatch.setattr(bc, "_chromium_has_cookie", lambda cf, d, n: True)
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["x.com"], "auth_token", source="X")
    msg = str(ei.value)
    assert "Multiple" in msg and "chrome/Default" in msg and "chrome/Profile 1" in msg
    assert "$OPYT_BROWSER" in msg


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


# ── Failure classification → actionable remediation ──────────────────────────────

def test_browser_launch_failure_is_not_reported_as_logged_out(monkeypatch):
    """A session Opyt found but could not decrypt because the browser would not start is
    NOT 'log in somewhere'. Telling the user to log in again would be a dead end."""
    _stub_reads(monkeypatch, installed=["chrome"],
                reads={("chrome", "Default"): {"sessionKey": "tok"}})
    monkeypatch.setattr(bc, "_read_one", lambda b, d, *, cookie_file=None: (
        {}, cdp.CDPError("browser exited (code 1) before opening DevTools")))
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    msg = str(ei.value)
    assert "could not start Chrome" in msg and "/Applications" in msg


def test_keychain_denied_remediation(monkeypatch):
    """Arc is the surviving Keychain backend — browser_cookie3 asks macOS for its Safe
    Storage key, which the transplant read never does."""
    err = Exception("Could not decrypt: user canceled keychain access")
    _stub_reads(monkeypatch, installed=["arc"], reads={("arc", None): err})
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["claude.ai"], "sessionKey", source="claude")
    msg = str(ei.value)
    assert "Keychain" in msg and ("Allow" in msg or "Always Allow" in msg)


def test_fda_needed_remediation(monkeypatch):
    err = PermissionError("Operation not permitted")
    _stub_reads(monkeypatch, installed=["safari"],
                reads={("safari", None): err})
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["substack.com"], "substack.sid", source="substack")
    msg = str(ei.value)
    assert "Full Disk Access" in msg


def test_blocked_read_is_not_mistaken_for_logged_out(monkeypatch):
    """The whole point: a blocked read must NOT collapse to the generic 'not logged in'
    — even when another browser is simply logged out."""
    err = PermissionError("Operation not permitted")
    _stub_reads(monkeypatch, installed=["safari", "chrome"],
                reads={("safari", None): err})  # chrome logged out, safari blocked
    with pytest.raises(SyncAuthError) as ei:
        bc.read_cookies(["substack.com"], "substack.sid", source="substack")
    assert "Full Disk Access" in str(ei.value)


def test_classify_prefers_backend_consent():
    assert bc._classify_failure(_backend("safari"), PermissionError("x")) == "fda_needed"
    assert bc._classify_failure(_backend("arc"),
                                Exception("keychain error")) == "keychain_denied"
    assert bc._classify_failure(_backend("arc"), Exception("weird")) == "other"
    # A transplant read classifies by TYPE — no Keychain is involved to deny.
    assert bc._classify_failure(_backend("chrome"), cdp.CDPError("boom")) == "browser_launch"
    assert bc._classify_failure(_backend("chrome"),
                                Exception("keychain error")) == "other"


# ── consent_prewarn copy per backend ─────────────────────────────────────────────

def test_consent_prewarn_copy():
    """Chromium reads raise no dialog any more, so there is nothing to pre-warn about."""
    assert bc.consent_prewarn(_backend("chrome")) is None
    assert bc.consent_prewarn(_backend("brave")) is None
    assert "Keychain" in bc.consent_prewarn(_backend("arc"))
    assert "Full Disk Access" in bc.consent_prewarn(_backend("safari"))
    assert bc.consent_prewarn(_backend("firefox")) is None
    assert bc.consent_prewarn(None) is None


# ── settings.yaml cookies.browser knob resolves ──────────────────────────────────

def test_settings_browser_knob(monkeypatch, tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("vault:\n  vault_path: /tmp/v\ncookies:\n  browser: edge\n")
    monkeypatch.setenv("OPYT_CONFIG", str(cfg))
    monkeypatch.delenv("OPYT_BROWSER", raising=False)
    from opyt_core.config import cookie_browser
    assert cookie_browser() == "edge"
    # env overrides settings
    monkeypatch.setenv("OPYT_BROWSER", "brave")
    assert cookie_browser() == "brave"
    # 'auto' means None
    monkeypatch.setenv("OPYT_BROWSER", "auto")
    assert cookie_browser() is None


# ── consent pre-warn copy (backend-derived — see guard `hardcoded-consent-prewarn`) ──

def test_fda_prewarn_carries_the_local_only_reassurance():
    """The 'nothing leaves your machine' sentence is doing real work in the one flow where a
    user decides whether to let an agent read their browser — BOTH consent branches carry it."""
    msg = bc.consent_prewarn(bc.backend_for("safari"))
    assert "Full Disk Access" in msg and "Nothing leaves your machine" in msg
    assert "machine" in bc.consent_prewarn(bc.backend_for("arc"))


def test_prewarn_installed_names_the_installed_browser(monkeypatch):
    """An Arc-only machine must be warned about Arc — never about a Chrome it doesn't have."""
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("arc")])
    msg = bc.prewarn_installed()
    assert "Arc" in msg and "Chrome" not in msg


def test_a_chromium_only_machine_gets_no_prewarn(monkeypatch):
    """The user-visible half of the change: on the common machine there is no native
    dialog left to warn about, so onboarding must not promise one."""
    monkeypatch.setattr(bc, "installed_backends",
                        lambda: [bc.backend_for("chrome"), bc.backend_for("brave")])
    assert bc.prewarn_installed() is None


# ── OPYT's own login profiles (created by guided_login) ──────────────────────────

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
