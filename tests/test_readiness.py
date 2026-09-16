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


# ── A trial key that stopped working is not an unfunded one ───────────────────────────────────
# The SAME 402 arrives for both, and the two remedies are opposite: one person tops up an
# account, the other does not have one to top up. Fold them together and the trial user is sent
# to a credits page for an account they never made — a dead end at the exact moment they were
# about to become a customer.

@pytest.mark.parametrize("ping,want", [
    ((False, "... HTTP 402 ...", 402), "trial_over"),   # the cap is spent
    ((False, "... HTTP 401 ...", 401), "trial_over"),   # `expires_at` passed, or revoked
    ((False, "... HTTP 403 ...", 403), "trial_over"),
    ((True,  "ok",               None), "ok"),          # a live trial is just a live key
    ((False, "connection reset", None), "unknown"),     # indeterminate stays indeterminate
])
def test_a_trial_key_has_its_own_end_state(monkeypatch, ping, want):
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_is_trial", lambda: True)
    monkeypatch.setattr(readiness, "_ping", lambda k: ping)
    assert readiness.openrouter()["state"] == want


def test_a_spent_trial_is_never_sent_to_a_credits_page(monkeypatch):
    """The load-bearing half. `TOPUP_URL` is correct advice for a user with an account and wrong
    advice for one without, so it must not reach this state's message."""
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_is_trial", lambda: True)
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "HTTP 402", 402))
    assert readiness.TOPUP_URL not in readiness.openrouter()["message"]


def test_a_user_key_keeps_the_advice_it_always_had(monkeypatch):
    """The trial branch must be reachable ONLY from a trial. A regression that keys it on
    anything else silently stops telling real customers how to fund their accounts."""
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_is_trial", lambda: False)
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "HTTP 402", 402))
    out = readiness.openrouter()
    assert out["state"] == "unfunded" and readiness.TOPUP_URL in out["message"]


def test_the_cost_is_stated_wherever_a_card_is(monkeypatch):
    """A user sent to a payment page is told the size of the amount. Asserted against the one
    constant rather than against a phrase, so correcting the number — see the ⚠️ on COST_NOTE —
    is a one-line edit that cannot leave a stale copy behind in a prompt."""
    monkeypatch.setattr(readiness, "_credential", lambda s: "sk-or-x")
    monkeypatch.setattr(readiness, "_is_trial", lambda: False)
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "HTTP 402", 402))
    assert readiness.COST_NOTE in readiness.openrouter()["message"]
