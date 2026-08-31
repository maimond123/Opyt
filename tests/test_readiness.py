"""
tests/test_readiness.py

Presence is NOT readiness. `keys.status()` answers
"is a key SET", which green-lights an OpenRouter key with no credit behind it — that key
authenticates fine and then fails every embed, so the store builds and can never be queried.

The load-bearing property here is that 401 and 402 stay APART. Fold them together and the
remedy inverts: a 402 treated as "auth failed, retry auth" mints a second unfunded key, then
a third, and the loop neither terminates nor fixes anything.
"""

import pytest

from opyt_core import readiness


@pytest.mark.parametrize("cred,ping,want", [
    (None,        None,                                    "missing"),
    ("sk-or-x",   (True,  "openrouter key is valid", None), "ok"),
    ("sk-or-x",   (False, "... HTTP 402 ...",        402),  "unfunded"),
    ("sk-or-x",   (False, "... HTTP 401 ...",        401),  "dead"),
    ("sk-or-x",   (False, "connection reset",        None), "unknown"),
])
def test_openrouter_states(monkeypatch, cred, ping, want):
    monkeypatch.setattr(readiness, "_credential", lambda s: cred)
    if ping is not None:
        monkeypatch.setattr(readiness, "_ping", lambda k: ping)
    assert readiness.openrouter()["state"] == want


def test_unfunded_message_does_not_tell_you_to_redo_oauth(monkeypatch):
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "HTTP 402", 402))
    msg = readiness.openrouter()["message"].lower()
    assert "add credit" in msg or "top up" in msg
    assert "oauth" not in msg and "re-run onboard" not in msg


def test_unknown_names_both_possibilities(monkeypatch):
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "connection reset", None))
    msg = readiness.openrouter()["message"].lower()
    assert "credit" in msg and "key" in msg      # both, because we cannot tell them apart


def test_no_key_value_is_ever_returned(monkeypatch):
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-SECRET")
    monkeypatch.setattr(readiness, "_ping", lambda k: (True, "ok", None))
    assert "SECRET" not in str(readiness.openrouter())
