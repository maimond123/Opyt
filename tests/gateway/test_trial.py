"""
tests/gateway/test_trial.py

The gateway half of the starter allowance — the half that spends the operator's money.

Every property here is a bound on a faucet:

  • ONE PER SUBJECT, and the subject comes from the POOL, never from the caller. A child that
    could name its own subject could claim an allowance per name it invented.
  • THE LEDGER SURVIVES A RESTART. It is the only durable thing this process writes, and if it
    were not, every deploy would hand everybody a fresh allowance.
  • RESERVE BEFORE MINTING. Two calls that arrive together must not both pass the check.
  • THE CALLBACK MUST BE LOOPBACK. The gateway is about to bounce a browser to a caller-supplied
    URL that will shortly receive a code buying a funded key.
"""
from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
from fastmcp.server.auth.providers.google import GoogleProvider
from starlette.testclient import TestClient

from gateway import trial
from gateway.app import build_app
from gateway.children import Child


class _Process:
    pid = 1
    returncode = None

    def terminate(self):
        self.returncode = 0

    async def wait(self):
        return 0


def _app(tmp_path, monkeypatch):
    async def no_token(self, token):
        return None
    monkeypatch.setattr(GoogleProvider, "verify_token", no_token)
    return build_app(base_url="https://gw.example.com", client_id="cid", client_secret="secret",
                     homes_root=tmp_path, idle_seconds=9999)


# ── the ledger ────────────────────────────────────────────────────────────────────────────────

def test_one_allowance_per_subject(tmp_path):
    ledger = trial.Ledger(tmp_path / "l.json")
    ledger.reserve("42", daily_cap=100)
    with pytest.raises(trial.TrialUnavailable, match="already_claimed"):
        ledger.reserve("42", daily_cap=100)


def test_the_ledger_survives_a_restart(tmp_path):
    """The one durable thing the gateway writes, and the reason it is durable: an in-memory
    ledger would re-open the faucet on every deploy."""
    trial.Ledger(tmp_path / "l.json").reserve("42", daily_cap=100)
    with pytest.raises(trial.TrialUnavailable, match="already_claimed"):
        trial.Ledger(tmp_path / "l.json").reserve("42", daily_cap=100)


def test_a_daily_cap_bounds_the_blast_radius(tmp_path):
    ledger = trial.Ledger(tmp_path / "l.json")
    ledger.reserve("a", daily_cap=2)
    ledger.reserve("b", daily_cap=2)
    with pytest.raises(trial.TrialUnavailable, match="daily_cap"):
        ledger.reserve("c", daily_cap=2)


def test_an_unreadable_ledger_refuses_rather_than_over_mints(tmp_path):
    """The one place `fail-safe` does NOT mean "degrade to empty": an empty read here is an
    unbounded faucet, so a ledger that cannot be trusted stops the mint instead."""
    path = tmp_path / "l.json"
    path.write_text("{not json")
    with pytest.raises(trial.TrialUnavailable, match="ledger_unreadable"):
        trial.Ledger(path).reserve("42", daily_cap=100)


def test_a_failed_mint_gives_the_slot_back(tmp_path, monkeypatch):
    """A reservation that never became a key must not cost that person their allowance — but
    only an UPSTREAM failure releases it. A refusal means they had their turn."""
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    ledger = trial.Ledger(tmp_path / "l.json")

    class _Http:
        async def post(self, *a, **kw):
            raise httpx.ConnectError("nope")

    import asyncio
    with pytest.raises(trial.TrialUnavailable, match="upstream"):
        asyncio.run(trial.mint(_Http(), "42", ledger))
    assert ledger.claimed("42") is None


def test_a_mint_is_capped_expiring_and_never_resetting(tmp_path, monkeypatch):
    """⚠️ `limit_reset` MUST NOT be sent. A daily-resetting cap turns every trial key into a
    permanent free tier against the operator's balance — a different, far more expensive
    product, arrived at by one extra field."""
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    sent = {}

    class _Http:
        async def post(self, url, **kw):
            sent.update(url=url, json=kw["json"], headers=kw["headers"])
            return httpx.Response(200, json={"key": "sk-or-v1-new", "data": {"hash": "h"}})

    import asyncio
    out = asyncio.run(trial.mint(_Http(), "42", trial.Ledger(tmp_path / "l.json")))

    assert "limit_reset" not in sent["json"]
    assert sent["json"]["limit"] > 0
    # ⚠️ `creator_user_id` MUST NOT be sent. OpenRouter answers `400 creator_user_id is only
    # valid for organization-owned keys`, and a personal account is what most operators run —
    # so sending it makes minting fail for exactly the common case. Attribution lives in the
    # NAME and in our own ledger, neither of which depends on the provider's account type.
    assert "creator_user_id" not in sent["json"]
    assert sent["json"]["name"].endswith("42")
    # ⚠️ THE SPELLING IS THE CONTRACT, and this assertion used to be `and expires_at` — merely
    # truthy, which the WRONG value satisfied perfectly. Live on 2026-09-11 the first real mint
    # returned `400 Invalid request field: expires_at` because Python's `isoformat()` emits
    # `+00:00` and OpenRouter accepts only `Z` for the identical instant. Every hosted user fell
    # through to the OpenRouter approval the allowance exists to avoid, and the suite was green.
    # Assert the shape a remote system actually parses, never just that a field was populated.
    assert sent["json"]["expires_at"].endswith("Z")
    assert "+00:00" not in sent["json"]["expires_at"]
    datetime.fromisoformat(sent["json"]["expires_at"])      # and it must read back
    assert out["key"] == "sk-or-v1-new"


def test_a_gateway_with_no_management_key_mints_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(trial.MANAGEMENT_KEY_ENV, raising=False)
    assert trial.enabled() is False
    import asyncio
    with pytest.raises(trial.TrialUnavailable, match="not_configured"):
        asyncio.run(trial.mint(None, "42", trial.Ledger(tmp_path / "l.json")))


# ── the open-redirect boundary ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://evil.example.com/cb/x",          # not loopback
    "http://evil.example.com/cb/x",           # not loopback, right scheme
    "http://127.0.0.1.evil.com/cb/x",         # a hostname that merely starts like one
    "http://localhost:5000/cb/x?code=stolen",  # smuggling a query past the far-end parse
    "notaurl",
])
def test_only_this_users_own_machine_may_receive_the_bounce(url):
    assert trial.loopback_callback(url) is None


@pytest.mark.parametrize("url", ["http://localhost:5000/cb/abc", "http://127.0.0.1:5000/cb/abc"])
def test_a_loopback_callback_is_allowed(url):
    assert trial.loopback_callback(url) == url


def test_pkce_binds_the_key_to_the_process_that_started_the_flow():
    verifier, challenge = __import__("opyt_core.local_auth", fromlist=["x"]).pkce_pair()
    assert trial.verify_pkce(verifier, challenge) is True
    assert trial.verify_pkce("some-other-verifier", challenge) is False


def test_a_subject_is_read_from_the_id_token_google_returned():
    import base64
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "42"}).encode()).rstrip(b"=").decode()
    assert trial.subject_from_id_token(f"header.{payload}.sig") == "42"
    assert trial.subject_from_id_token("garbage") is None


# ── the hosted door ───────────────────────────────────────────────────────────────────────────

def test_the_hosted_mint_reads_the_subject_from_the_pool_not_the_request(tmp_path, monkeypatch):
    """THE load-bearing authorization property. The child proves which child it is; the gateway
    decides who that is. A child permitted to name its own subject could mint per invented name.
    """
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    seen = {}

    async def fake_mint(http, subject, ledger):
        seen["subject"] = subject
        return {"key": "sk-or-v1-new", "hash": "h", "limit": 0.25, "expires_at": "z"}
    monkeypatch.setattr(trial, "mint", fake_mint)

    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        child = Child(subject="realsubject", home=tmp_path / "realsubject", port=5432,
                      proc=_Process(), last_seen=0, interaction_key="child-key")  # type: ignore[arg-type]
        app.state.pool._children[child.subject] = child

        out = client.post("/_internal/hosted/trial",
                          headers={"X-Opyt-Hosted-Interaction-Key": "child-key"},
                          json={"subject": "attacker-chosen"})

    assert out.status_code == 200
    assert seen["subject"] == "realsubject"


def test_an_unproven_caller_gets_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        out = client.post("/_internal/hosted/trial",
                          headers={"X-Opyt-Hosted-Interaction-Key": "guessed"})
    assert out.status_code == 404


def test_a_refusal_reads_as_settled_not_as_retry(tmp_path, monkeypatch):
    """409, not 500. Every refusal is a final answer about this subject, and the child must take
    the other path rather than try again."""
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")

    async def refuse(http, subject, ledger):
        raise trial.TrialUnavailable("already_claimed")
    monkeypatch.setattr(trial, "mint", refuse)

    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        child = Child(subject="42", home=tmp_path / "42", port=5432,
                      proc=_Process(), last_seen=0, interaction_key="child-key")  # type: ignore[arg-type]
        app.state.pool._children[child.subject] = child
        out = client.post("/_internal/hosted/trial",
                          headers={"X-Opyt-Hosted-Interaction-Key": "child-key"})
    assert out.status_code == 409
    assert out.json()["error"] == "already_claimed"


def test_a_child_never_inherits_the_management_key(tmp_path, monkeypatch):
    """⚠️ THE WORST THING ON THE BOX. Every other gateway secret proves an identity; this one
    mints spend against the operator's balance. A hosted child asks the gateway to mint and
    receives one capped key — it has no use for the key that makes keys, so it never sees it.

    The ledger path is stripped for the same reason: a child able to write it could delete its
    own row and claim another allowance, which is the cap defeated from inside.
    """
    from gateway.children import _child_env

    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    monkeypatch.setenv("OPYT_TRIAL_LEDGER", "/srv/homes/.trial-ledger.json")

    env = _child_env(tmp_path / "42", "42",
                     interaction_registration_url="http://127.0.0.1:8080/_internal/hosted/register",
                     interaction_url="https://gw.example.com",
                     interaction_key="child-key",
                     interaction_trial_url="http://127.0.0.1:8080/_internal/hosted/trial")

    assert trial.MANAGEMENT_KEY_ENV not in env
    assert "OPYT_TRIAL_LEDGER" not in env
    assert "sk-or-v1-management" not in json.dumps(dict(env))
    # What it DOES get: the address it asks for a mint at, and the key proving which child it is.
    assert env["OPYT_HOSTED_INTERACTION_TRIAL_URL"].endswith("/_internal/hosted/trial")


def test_a_gateway_that_cannot_mint_does_not_advertise_a_trial_to_its_children(tmp_path):
    """Fail-safe, at the env boundary: without the URL the child's `trial.hosted_enabled` is
    False, so `onboard` offers the OpenRouter approval instead of a step that can only fail."""
    from gateway.children import _child_env

    env = _child_env(tmp_path / "42", "42",
                     interaction_registration_url="http://127.0.0.1:8080/_internal/hosted/register",
                     interaction_url="https://gw.example.com",
                     interaction_key="child-key",
                     interaction_trial_url=None)

    assert "OPYT_HOSTED_INTERACTION_TRIAL_URL" not in env


# ── the operator's audit ──────────────────────────────────────────────────────────────────────
# This is where `readiness.COST_NOTE` stops being an intention. It is also the sweeper, because
# one call to OpenRouter answers both questions: what has this key spent, and can it still spend.

class _AuditHttp:
    """Answers `GET /api/v1/keys/{hash}` from a dict, and records every PATCH."""

    def __init__(self, by_hash: dict, fail: set[str] | None = None):
        self.by_hash = by_hash
        self.fail = fail or set()
        self.patched: list[str] = []

    async def get(self, url, **kw):
        key_hash = url.rsplit("/", 1)[1]
        if key_hash in self.fail:
            raise httpx.ConnectError("nope")
        if key_hash not in self.by_hash:
            return httpx.Response(404, json={})
        return httpx.Response(200, json={"data": self.by_hash[key_hash]})

    async def patch(self, url, **kw):
        self.patched.append(url.rsplit("/", 1)[1])
        return httpx.Response(200, json={})


def _ledger_with(tmp_path, rows: dict) -> trial.Ledger:
    (tmp_path / "l.json").write_text(json.dumps(rows))
    return trial.Ledger(tmp_path / "l.json")


def test_the_audit_reads_real_spend_and_never_a_key_value(tmp_path, monkeypatch):
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    ledger = _ledger_with(tmp_path, {
        "42": {"at": 0, "hash": "h42", "limit": 0.25, "expires_at": "2099-01-01T00:00:00+00:00"},
    })
    http = _AuditHttp({"h42": {"usage": 0.031, "limit_remaining": 0.219, "disabled": False}})

    import asyncio
    rows = asyncio.run(trial.audit(http, ledger))

    assert rows[0]["subject"] == "42" and rows[0]["usage"] == 0.031
    assert rows[0]["state"] == "live"
    assert "sk-or-v1" not in json.dumps(rows)


@pytest.mark.parametrize("data,expires,want", [
    ({"usage": 0.25, "limit_remaining": 0, "disabled": False},
     "2099-01-01T00:00:00+00:00", "spent"),
    ({"usage": 0.01, "limit_remaining": 0.24, "disabled": False},
     "2000-01-01T00:00:00+00:00", "expired"),
    ({"usage": 0.01, "limit_remaining": 0.24, "disabled": True},
     "2099-01-01T00:00:00+00:00", "off"),
    ({"usage": 0.01, "limit_remaining": 0.24, "disabled": False},
     "2099-01-01T00:00:00+00:00", "live"),
])
def test_spent_and_expired_are_reported_apart(data, expires, want):
    """A user meets both as one `trial_over`, but the operator must not: a pile of expired-and-
    unspent keys is a product signal (claimed, never used), while spent is the thing working."""
    assert trial._state_of(data, {"expires_at": expires}) == want


def test_a_row_the_provider_does_not_know_is_shown_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    ledger = _ledger_with(tmp_path, {"42": {"at": 0, "hash": "gone", "limit": 0.25},
                                     "77": {"at": 0, "hash": None, "limit": 0.25}})
    import asyncio
    rows = asyncio.run(trial.audit(_AuditHttp({}), ledger))

    assert {r["subject"]: r["state"] for r in rows} == {"42": "missing", "77": "unfilled"}


def test_unused_keys_do_not_drag_the_average_down(tmp_path):
    """⚠️ THE NUMBER THAT BACKS A PROMISE TO USERS. A mint nobody used costs $0, so averaging
    those zeros in understates what a real user costs by however many tyre-kickers signed up —
    which is the wrong direction for a string shown beside a card field."""
    summary = trial.cost_summary([
        {"usage": 0.08}, {"usage": 0.12}, {"usage": 0.0}, {"usage": 0.0}, {"usage": None},
    ])
    assert summary["used"] == 2 and summary["unused"] == 2
    assert summary["mean_usd"] == 0.1          # not 0.05
    assert summary["total_usd"] == 0.2


def test_a_sweep_never_touches_a_live_key(tmp_path, monkeypatch):
    """Housekeeping, not enforcement. Disabling a spent key changes nothing for its holder;
    disabling a live one cuts somebody off mid-use."""
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    ledger = _ledger_with(tmp_path, {
        "spent": {"at": 0, "hash": "h1", "limit": 0.25, "expires_at": "2099-01-01T00:00:00+00:00"},
        "live": {"at": 0, "hash": "h2", "limit": 0.25, "expires_at": "2099-01-01T00:00:00+00:00"},
    })
    http = _AuditHttp({"h1": {"usage": 0.25, "limit_remaining": 0, "disabled": False},
                       "h2": {"usage": 0.01, "limit_remaining": 0.24, "disabled": False}})

    import asyncio
    rows = asyncio.run(trial.audit(http, ledger))
    sweepable = [r["hash"] for r in rows if r["state"] in ("spent", "expired")]

    assert sweepable == ["h1"]
    asyncio.run(trial.disable(http, "h1"))
    assert http.patched == ["h1"]


def test_an_audit_needs_a_management_key(tmp_path, monkeypatch):
    monkeypatch.delenv(trial.MANAGEMENT_KEY_ENV, raising=False)
    import asyncio
    with pytest.raises(trial.TrialUnavailable, match="not_configured"):
        asyncio.run(trial.audit(None, trial.Ledger(tmp_path / "l.json")))


def test_a_rejected_management_key_is_not_reported_as_a_network_problem(tmp_path, monkeypatch):
    """Opposite remedies, so they must not share a word. An operator who typo'd their management
    key and reads "unreachable" on every row goes and checks whether OpenRouter is down."""
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "not-a-provisioning-key")

    class _Rejects:
        async def get(self, url, **kw):
            return httpx.Response(401, json={"error": "unauthorized"})

    ledger = _ledger_with(tmp_path, {"42": {"at": 0, "hash": "h42", "limit": 0.25}})
    import asyncio
    rows = asyncio.run(trial.audit(_Rejects(), ledger))

    assert rows[0]["state"] == "unauthorized"
