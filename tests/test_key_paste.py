"""
tests/test_key_paste.py

The paste-over-loopback path, for keys that have no OAuth flow. The properties that matter:
a validated key is written, an INVALID key is refused BEFORE it reaches disk, and the value
never appears in the returned payload.
"""

import pytest

from opyt_core import key_paste

# A real registry row, so the fixture stays honest about what this module is for. It used to be
# `twitterapi` / `TWITTERAPI_KEY`; that credential was deleted on 2026-08-30 with its provider,
# and the guard `retired-twitterapi-credential` bans the name outright. The paste flow itself is
# credential-agnostic, which is why swapping the row changes nothing these tests assert.
#
# ⚠️ `key_paste` has NO production caller as of 2026-08-30 — the twitterapi gate in
# `onboard_tools._phase_keys` was its only one, and OpenRouter uses OAuth. It is kept because
# GITHUB_TOKEN and S2_API_KEY have no acquisition path yet and this is the shape one would take.
SIGNUP_URL = "https://github.com/settings/tokens"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """`acquire()` now pauses `SIGNUP_FOCUS_DELAY_S` between the two browser opens (the tab-focus
    fix). Autouse so every test in this file stays instant by default; the two tests that assert
    on the delay itself override this with their own `monkeypatch.setattr`, which wins because it
    runs later in the same fixture stack."""
    monkeypatch.setattr(key_paste, "_default_sleep", lambda seconds: None)


class _Cap:
    url = "http://localhost:1/paste/n"

    def __init__(self, value):
        self._v = value

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def wait(self):
        return {"params": {}, "form": {"value": self._v}} if self._v else None


def test_valid_key_is_written(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap("abcdefghijkl"))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: True)
    monkeypatch.setattr(key_paste.credentials, "validate_credential",
                        lambda s, k: (True, "ok"))
    out = key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)
    assert out["status"] == "stored"
    assert "abcdefghijkl" not in str(out)          # ⚠️ never echo the value
    assert "GITHUB_TOKEN=abcdefghijkl" in (tmp_path / ".env").read_text()


def test_invalid_key_is_refused_and_not_written(monkeypatch, tmp_path):
    """⚠️ A bad key ON DISK is worse than no key: every later failure then looks like an
    outage instead of a setup mistake."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap("x"))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: True)
    monkeypatch.setattr(key_paste.credentials, "validate_credential",
                        lambda s, k: (False, "Key is too short"))
    out = key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)
    assert out["status"] == "invalid"
    assert not (tmp_path / ".env").exists()


def test_timeout_degrades_to_the_signup_url(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap(None))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: False)
    out = key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)
    assert out["status"] == "waiting"
    assert out["get_your_key_at"] == SIGNUP_URL


def test_the_waiting_message_tells_you_not_to_paste_into_chat(monkeypatch, tmp_path):
    """The whole channel rule is worthless if the user helpfully pastes it here anyway."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap(None))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: False)
    out = key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)
    assert "chat" in out["message"].lower()


# ── tab-focus race ────────────────────────────────────────────────────────────────
#
# THE BUG (found live, 2026-08-16): `acquire()` opens the signup URL, then IMMEDIATELY opens the
# local paste form. Both `webbrowser.open()` calls land within the same tick, and the browser
# gives focus to whichever tab opened SECOND — so the paste form (opened last) covers the signup
# page (opened first), and a user sees only "paste your key" with no visible way to go get one.
# The signup tab is still there, one tab over; nobody notices it's there to look for it.


def test_signup_opens_before_the_paste_form(monkeypatch, tmp_path):
    """Ordering alone, no timing — the cheap half of the property."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap("abcdefghijkl"))
    monkeypatch.setattr(key_paste.credentials, "validate_credential", lambda s, k: (True, "ok"))
    opened = []
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: opened.append(u) or True)

    key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)

    assert opened == [SIGNUP_URL, "http://localhost:1/paste/n"]


def test_a_delay_separates_the_two_opens_so_the_signup_tab_keeps_focus(monkeypatch, tmp_path):
    """THE ACTUAL FIX. Without a pause between the two `webbrowser.open()` calls, both land in the
    same tick and the browser gives focus to whichever opened second — the paste form, which is
    exactly backwards from how a human wants to be walked through this: go get the key, THEN come
    back and paste it.

    Injected via `_default_sleep`, matching `frontier_execute._default_sleep` — a real test must
    never actually block on wall-clock time."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap("abcdefghijkl"))
    monkeypatch.setattr(key_paste.credentials, "validate_credential", lambda s, k: (True, "ok"))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: True)
    calls = []
    monkeypatch.setattr(key_paste, "_default_sleep", lambda s: calls.append(s))

    key_paste.acquire("github", "GITHUB_TOKEN", signup_url=SIGNUP_URL)

    assert calls == [key_paste.SIGNUP_FOCUS_DELAY_S]
    assert key_paste.SIGNUP_FOCUS_DELAY_S > 0


def test_no_signup_url_means_no_delay(monkeypatch, tmp_path):
    """Nothing to let keep focus over, so no reason to pause — OpenRouter's OAuth flow has no
    signup_url and must not pick up a pointless wait."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(key_paste.local_auth, "Capture", lambda **kw: _Cap("abcdefghijkl"))
    monkeypatch.setattr(key_paste.credentials, "validate_credential", lambda s, k: (True, "ok"))
    monkeypatch.setattr(key_paste, "_open_browser", lambda u: True)
    calls = []
    monkeypatch.setattr(key_paste, "_default_sleep", lambda s: calls.append(s))

    key_paste.acquire("github", "GITHUB_TOKEN", signup_url="")

    assert calls == []
