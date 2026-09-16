"""The notice that ends the silence measured at 05:08 on 2026-09-15 — a starter allowance spent,
four subsystems degrading correctly and in private, and a user who found out the next morning
from symptoms and blamed their X login.

The fixtures mirror the live hosted home the failure was measured in: an `openrouter` breaker
OPEN on repeated `403 Key limit exceeded`, and a trial marker whose ceiling is $0.10 — the one
that makes a hardcoded "20 cents" wrong.
"""
import time

import pytest

from opyt_core import readiness
from pipeline.circuit_breaker import CircuitBreaker
from pipeline.kb import allowance_notice as an

_KEY_LIMIT = '_BackendError: HTTP 403: {"error":{"message":"Key limit exceeded (total limit)"}}'


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """A home of its own, and a cold verdict cache. The cache is module state by design — it is
    what keeps a repeating notice from adding an HTTP round trip to every degraded call — so a
    test that did not clear it would read the previous test's answer."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(an, "_CACHE", None)
    return tmp_path


def _trip(service: str, *, detail: str = _KEY_LIMIT) -> CircuitBreaker:
    """Open a breaker the way the provider does: consecutive failures up to the threshold."""
    breaker = CircuitBreaker(service)
    for _ in range(breaker.threshold):
        breaker.record_failure(detail)
    return breaker


def _verdict(monkeypatch, state: str, message: str = "the message readiness wrote") -> list:
    """Stand in for the paid probe, and COUNT it — the call this design exists to spend only
    when something is actually wrong."""
    calls: list = []

    def fake():
        calls.append(1)
        return {"state": state, "message": message}

    monkeypatch.setattr(readiness, "openrouter", fake)
    return calls


# ── the trigger ───────────────────────────────────────────────────────────────────────────────

def test_a_healthy_install_is_silent_and_pays_for_nothing(monkeypatch):
    """The common path, and the whole argument for a breaker trigger over a probe on every call:
    the answer is "no" almost always, and it must be free."""
    calls = _verdict(monkeypatch, "ok")
    assert an.allowance_notice() is None
    assert calls == [], "a healthy install must not spend a network round trip to learn it"


def test_an_open_breaker_past_its_cooldown_is_still_blocked():
    """⚠️ THE TRAP `provider_blocked` EXISTS TO AVOID. `CircuitBreaker.peek()` answers "would a
    call be permitted", which goes True the moment the 60s cooldown elapses — so a key that is
    dead forever reads as fine for all but the first minute after each trip. Measured on the live
    fixture: the `openrouter` row sat OPEN with 14 consecutive 403s, eight minutes past
    `opened_at`. The raw `state` column is the durable fact."""
    breaker = _trip("openrouter")
    assert breaker.peek() is False
    # Wind the clock past the cooldown, exactly as eight real minutes would.
    import sqlite3
    conn = sqlite3.connect(str(breaker._db_path))
    conn.execute("UPDATE circuit_breaker SET opened_at=? WHERE service='openrouter'",
                 (time.time() - 600,))
    conn.commit()
    conn.close()

    assert breaker.peek() is True, "precondition: peek() is the misleading read"
    assert readiness.provider_blocked() == "openrouter"


def test_the_embed_circuit_counts_too():
    """Both halves of the model provider block the product, and the live fixture had both open."""
    _trip("openrouter-embed", detail='EmbedError: HTTP 403: "Key limit exceeded (total limit)"')
    assert readiness.provider_blocked() == "openrouter-embed"


def test_a_handles_own_trouble_is_not_the_providers(monkeypatch):
    """An `oracle-refresh:*` breaker is open for ONE handle's repeated failures and says nothing
    about the key. Firing on it would put a top-up notice on a healthy account — which is the
    inverse of `c7979b64`, where a provider outage was charged to a handle."""
    _trip("oracle-refresh:x:user:975243637:x")
    calls = _verdict(monkeypatch, "trial_over")
    assert readiness.provider_blocked() is None
    assert an.allowance_notice() is None
    assert calls == []


def test_an_unreadable_breaker_table_invents_no_outage(monkeypatch):
    """Fail-safe, in the quiet direction. A notice we cannot justify is worse than the silence
    this whole module is fixing."""
    def boom():
        raise OSError("disk gone")

    monkeypatch.setattr("pipeline.circuit_breaker.status", boom)
    assert readiness.provider_blocked() is None


# ── the confirmation ──────────────────────────────────────────────────────────────────────────

def test_a_spent_allowance_fires(monkeypatch):
    """THE MEASURED FAILURE, inverted."""
    _trip("openrouter")
    _verdict(monkeypatch, "trial_over", "The starter allowance that came with OPYT is used up.")

    notice = an.allowance_notice()

    assert notice["state"] == "trial_over"
    assert notice["circuit"] == "openrouter"
    assert "used up" in notice["reason"]
    assert "SAY THIS NOW" in notice["host_note"]


def test_an_outage_is_not_an_exhausted_key(monkeypatch):
    """The breaker opens for ANY repeated provider failure, so it is a trigger and not an answer.
    When OpenRouter itself is down, `readiness`'s probe fails for the same reason and returns
    `unknown` — and telling a user with a funded account to go top it up, on the one day the
    provider is having a bad time, is the false positive this branch prevents."""
    _trip("openrouter", detail="_BackendError: HTTP 502: bad gateway")
    _verdict(monkeypatch, "unknown", "Could not verify the starter allowance (timeout).")

    assert an.allowance_notice() is None


def test_the_probe_is_paid_for_once_per_window(monkeypatch):
    """The notice REPEATS — it announces a state, not an event — so without a memo every degraded
    call would add an HTTP round trip to a call that is already failing slowly."""
    _trip("openrouter")
    calls = _verdict(monkeypatch, "trial_over")

    for _ in range(5):
        assert an.allowance_notice() is not None
    assert len(calls) == 1


def test_a_non_answer_is_not_memoed_like_an_answer(monkeypatch):
    """⚠️ `unknown` MEANS THE PROBE COULD NOT RUN, and the usual reason it could not run is the
    breaker the failing key just opened. Held for the full window that non-answer silenced the
    notice for two minutes — measured 2026-09-15: `unknown` at t+0, first `dead` only at t+120 —
    across exactly the first calls a user makes after their key dies.

    A short memo, not none: when the breaker is open the re-probe costs no socket, but `unknown`
    also covers a real network failure, and retrying a timeout on every degraded call would put
    that latency on answers that are already slow.
    """
    assert an._UNKNOWN_TTL_S < an._VERDICT_TTL_S

    _trip("openrouter")
    calls = _verdict(monkeypatch, "unknown")
    assert an.allowance_notice() is None            # unknown never fires — it may be an outage
    assert len(calls) == 1

    # Inside the short window the non-answer is reused; past it, the probe is paid for again.
    an.allowance_notice()
    assert len(calls) == 1
    an._CACHE = (time.time() - 1, {"state": "unknown"})
    an.allowance_notice()
    assert len(calls) == 2, "an expired non-answer must be re-probed, not trusted"


def test_fixing_it_stops_the_notice_without_waiting_out_the_memo(monkeypatch):
    """A working key closes the breaker on its first success, and a closed breaker returns before
    the cache is ever consulted. So the user who acts is not told they are still broken."""
    breaker = _trip("openrouter")
    _verdict(monkeypatch, "trial_over")
    assert an.allowance_notice() is not None

    breaker.record_success()
    assert an.allowance_notice() is None


# ── the remedy ────────────────────────────────────────────────────────────────────────────────

def test_a_trial_user_is_sent_to_their_own_account_never_to_a_credits_page(monkeypatch):
    """Sending somebody to fund an account they have not made is a dead end — the rule
    `opyt_core/trial.py` was written around."""
    _trip("openrouter")
    _verdict(monkeypatch, "trial_over")

    notice = an.allowance_notice()

    assert notice["next_call"] == "onboard(start='openrouter')"
    assert "top_up_url" not in notice
    assert readiness.TOPUP_URL not in notice["reason"]


def test_a_user_key_with_no_credit_gets_the_top_up_page(monkeypatch):
    """The opposite person. They own the account, so funding it IS the remedy."""
    _trip("openrouter")
    _verdict(monkeypatch, "unfunded", f"...no credit... Add credit at {readiness.TOPUP_URL}")

    notice = an.allowance_notice()

    assert notice["top_up_url"] == readiness.TOPUP_URL
    assert "next_call" not in notice


def test_a_rejected_key_is_re_approved_not_paid_for(monkeypatch):
    """`dead` is a 401/403 on a user's own key. Paying would not fix it, and minting against an
    empty balance is the loop this mapping avoids."""
    _trip("openrouter")
    _verdict(monkeypatch, "dead", "Your stored OpenRouter key was rejected.")

    notice = an.allowance_notice()

    assert notice["next_call"] == "onboard(start='openrouter')"
    assert "top_up_url" not in notice


def test_no_amount_is_named_anywhere_in_the_notice(monkeypatch):
    """D4, structurally. `DEFAULT_LIMIT_USD` moved 0.10 → 0.20 on 2026-09-15 and existing keys
    keep the ceiling they were minted with — the live fixture holds $0.10 — so any figure written
    into this copy is wrong for some real user today and wrong again at the next raise. The
    sentence is rendered from `readiness`, which names none."""
    _trip("openrouter")
    _verdict(monkeypatch, "trial_over", "The starter allowance that came with OPYT is used up.")

    rendered = " ".join(str(v) for v in an.allowance_notice().values())

    for figure in ("20 cents", "10 cents", "$0.20", "$0.10", "0.20", "0.10", "20¢", "10¢"):
        assert figure not in rendered
