"""
tests/test_trial.py

The client half of the starter allowance. Two properties carry it, and both are about what
happens when something goes WRONG rather than when it works:

  • THE MARKER IS WRITTEN BEFORE THE KEY. A crash between the two writes must never leave a key
    that reads as the user's own, because that is the one state where OPYT tells somebody to
    fund an account they never made. The other order fails safe.
  • STORING A USER'S OWN KEY CLEARS THE MARKER. Both homes store through one helper precisely so
    this cannot be half-done — miss it and a funded personal account inherits the trial's copy
    the first time it runs dry, months later, with nothing on screen to explain why.
"""

import json

import pytest

from opyt_core import keys, openrouter_oauth, trial


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    return tmp_path


def test_no_marker_means_the_key_is_the_users_own(home):
    assert trial.is_trial() is False


def test_recording_a_trial_writes_no_secret(home):
    trial.record({"key": "sk-or-v1-SECRET", "hash": "abc123", "limit": 0.25,
                  "expires_at": "2026-10-11T00:00:00+00:00"})
    raw = trial.marker_path().read_text()
    assert "SECRET" not in raw
    assert json.loads(raw)["hash"] == "abc123"
    assert trial.is_trial() is True


def test_the_marker_is_written_before_the_key(home, monkeypatch):
    """Ordering, asserted by making the SECOND write fail. What survives a crash between them
    must be a marker with no key — which reads as `missing` and sends the user down the approval
    path — and never a key with no marker, which reads as their own."""
    def boom(*a, **kw):
        raise OSError("disk went away")
    monkeypatch.setattr(trial.keys, "set_key", boom)

    with pytest.raises(OSError):
        trial._store({"key": "sk-or-v1-x", "hash": "h", "limit": 0.25, "expires_at": "z"})

    assert trial.is_trial() is True                      # the marker landed
    assert not keys.env_path().exists()                  # the key did not


def test_storing_a_users_own_key_retires_the_marker(home):
    """Reached from BOTH homes: the hosted flow stores inside `_HostedApproval.complete`, the
    local one at the end of `acquire`, and both go through `_store_user_key`."""
    trial.record({"hash": "h", "limit": 0.25, "expires_at": "z"})
    assert trial.is_trial() is True

    openrouter_oauth._store_user_key("sk-or-v1-theirs")

    assert trial.is_trial() is False
    assert "sk-or-v1-theirs" in keys.env_path().read_text()


def test_a_home_that_had_its_allowance_is_not_offered_another(home):
    trial.record({"hash": "h", "limit": 0.25, "expires_at": "z"})
    assert trial.available() is False


def test_an_unreadable_marker_does_not_stop_setup(home):
    """Fail-safe: the worst a corrupt marker may cost is one wrong sentence at the moment the
    key dies, never a user who cannot set OPYT up at all."""
    trial.marker_path().write_text("{not json")
    assert trial.read_marker() is None
    assert trial.is_trial() is False


def test_acquire_never_returns_a_credential(home, monkeypatch):
    monkeypatch.setenv(trial.HOSTED_TRIAL_URL_ENV, "http://127.0.0.1:1/_internal/hosted/trial")
    monkeypatch.setattr(trial, "_post_json",
                        lambda *a, **kw: {"key": "sk-or-v1-SECRET", "hash": "h",
                                          "limit": 0.25, "expires_at": "z"})
    out = trial.acquire()
    assert out["status"] == "stored"
    assert "SECRET" not in str(out)


def test_a_gateway_refusal_is_soft(home, monkeypatch):
    """A trial that cannot be claimed must leave the caller able to offer the ordinary path, so
    it reports `unavailable` and never raises."""
    monkeypatch.setenv(trial.HOSTED_TRIAL_URL_ENV, "http://127.0.0.1:1/_internal/hosted/trial")

    def refuse(*a, **kw):
        raise trial.TrialError("already_claimed")
    monkeypatch.setattr(trial, "_post_json", refuse)

    assert trial.acquire()["status"] == "unavailable"


def test_hosted_is_advertised_only_when_the_gateway_wired_it(home, monkeypatch):
    """Separate from `openrouter_oauth.hosted_enabled` on purpose: an operator can route OAuth
    callbacks and hold no management key, and that gateway must not offer a mint it cannot do."""
    monkeypatch.delenv(trial.HOSTED_TRIAL_URL_ENV, raising=False)
    monkeypatch.setenv("OPYT_HOSTED_OPENROUTER", "1")
    assert trial.hosted_enabled() is False
