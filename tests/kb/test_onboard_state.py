"""
tests/kb/test_onboard_state.py

Where onboarding stands, recomputed from disk on every call. NO PHASE STATE FILE —
`pipeline/onboarding.py` held one and was deleted 2026-08-14 with phases named `vault_setup`
and `taxonomy`, both retired, so a stale phase file would have told a user they were done.

The one marker that survives is `onboard_consent_asked`, and the test named
`test_answered_no_to_both_is_NOT_re_asked` is the entire reason it exists.
"""

import pytest

from pipeline.kb import onboard_state


@pytest.fixture()
def _ready(monkeypatch, kb_home):
    """Everything green except the phase under test — each test knocks one thing out."""
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "ok", "message": ""})
    monkeypatch.setattr(onboard_state, "_x_profiles", lambda: ([{"profile": "Default",
                                                                "label": "you"}], None))
    (kb_home / "onboard_consent_asked").touch()
    monkeypatch.setattr(onboard_state, "_curation_ok", lambda: True)
    return kb_home


def test_all_green_is_done(_ready):
    assert onboard_state.derive()["phase"] == "done"


def test_missing_openrouter_is_keys(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "missing", "message": ""})
    assert onboard_state.derive()["phase"] == "keys"


def test_unfunded_openrouter_is_ALSO_keys(_ready, monkeypatch):
    """⚠️ decision 9: an unfunded account blocks, it does not pass through to phase 3."""
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: {"state": "unfunded", "message": ""})
    assert onboard_state.derive()["phase"] == "keys"


def test_no_browser_session_is_browser(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state, "_x_profiles", lambda: ([], None))
    assert onboard_state.derive()["phase"] == "browser"


def test_never_asked_is_consent_even_with_no_markers(_ready):
    (_ready / "onboard_consent_asked").unlink()
    assert onboard_state.derive()["phase"] == "consent"


def test_answered_no_to_both_is_NOT_re_asked(_ready):
    """The whole reason the `asked` marker exists."""
    assert not (_ready / "bookmark_catchup_consent").exists()
    assert not (_ready / "oracle_refresh_consent").exists()
    assert onboard_state.derive()["phase"] == "done"


def test_no_curation_run_is_curation(_ready, monkeypatch):
    monkeypatch.setattr(onboard_state, "_curation_ok", lambda: False)
    assert onboard_state.derive()["phase"] == "curation"


def test_probe_browser_false_does_not_read_cookies(_ready, monkeypatch):
    """⚠️ Reading cookies trips the macOS Keychain dialog. The pre-warn has to be rendered
    WITHOUT triggering the prompt it is warning about."""
    reads = []
    monkeypatch.setattr(onboard_state, "_x_profiles",
                        lambda: reads.append(1) or ([], None))
    onboard_state.derive(probe_browser=False)
    assert reads == []


def test_derive_never_raises_when_everything_is_broken(monkeypatch, kb_home):
    for name in ("_x_profiles", "_curation_ok"):
        monkeypatch.setattr(onboard_state, name, lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(onboard_state.readiness, "openrouter",
                        lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert onboard_state.derive()["phase"] == "keys"      # fail-safe: fall back to the start


def test_consent_markers_honor_the_rails_env_override(_ready, monkeypatch, tmp_path):
    """⚠️ Both rails resolve their marker through an env override. Re-deriving the path here
    would be a second source of truth that disagrees the moment an override is set."""
    marker = tmp_path / "elsewhere" / "bm-consent"
    marker.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPYT_BOOKMARK_CATCHUP_CONSENT", str(marker))
    assert onboard_state.derive()["consent"]["bookmark"] is False
    marker.touch()
    assert onboard_state.derive()["consent"]["bookmark"] is True
