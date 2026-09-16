"""The Substack saved-posts rail — consent, the session gate, single-flight, fail-safe.

Everything here fails SILENTLY in production, which is why it is asserted rather than watched.
The defect this rail closes is exactly that shape: `sync_substack_saved` has been written,
tested and correct since 2026-08, reachable only from a hand-run CLI, so the saved posts of every
user who never typed that command have never entered the store — and nothing anywhere said so.

The one property that is NOT a copy of `bookmark_catchup`'s is `consented()`. That rail accepts
an established store as consent, to grandfather users it was already running for. This one has no
such population, so the same clause would arm a brand-new metered import on every existing home
with the question never having been put.
"""
from __future__ import annotations

import pytest

from pipeline.kb import substack_saved_catchup as ssc


@pytest.fixture()
def rail_home(kb_home, monkeypatch):
    monkeypatch.delenv("OPYT_SUBSTACK_SAVED_CATCHUP_CONSENT", raising=False)
    # Since 2026-09-13 this rail gates on the OPYT-managed Substack session (`readable()` reads
    # only the managed profile now, no user-browser fallback), so a mechanics test needs one
    # present or the pass returns `no_session` before reaching what it is about. The gate has its
    # own tests below, which set this to False deliberately.
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: True)
    return kb_home


@pytest.fixture()
def no_spend(monkeypatch):
    """Stub the two things a pass would otherwise pay for, and hand back the call log so a test
    can assert that NOTHING paid ran."""
    calls = []
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_curation.sync_substack_saved",
                        lambda *a, **kw: calls.append(kw) or {"source": "substack-saved",
                                                              "added": 0})
    return calls


# ── consent ─────────────────────────────────────────────────────────────────────
def test_unconsented_returns_needs_consent_and_spends_nothing(rail_home, no_spend):
    out = ssc.run_substack_saved_catchup()
    assert out["status"] == "needs_consent"
    assert no_spend == []


def test_a_caller_that_granted_consent_then_runs_proceeds(rail_home, no_spend):
    """The path `onboard._apply_consent` takes: ask, grant, THEN queue. Consent is read here and
    granted by the surface that asked — a `force` that granted it on the caller's behalf would
    make one word mean both "run now" and "the user agreed"."""
    assert ssc.run_substack_saved_catchup()["status"] == "needs_consent"
    ssc.grant_consent()
    assert ssc.consented() is True
    assert ssc.run_substack_saved_catchup()["status"] == "ok"


def test_an_established_store_is_NOT_auto_consented(rail_home, no_spend):
    """⚠️ THE ONE DELIBERATE DIFFERENCE FROM THE OTHER RAILS, and the reason is the whole point
    of this file.

    `bookmark_catchup` and `curation_catchup` treat stored atoms as consent, because both were
    already running before their marker existed and the clause spares an existing user a question
    about something months old. Nothing has ever run `sync_substack_saved` unattended, so no such
    population exists — the clause would instead switch on a brand-new metered import for every
    home that already holds content, the first time a worker reached it, with the question never
    having been put. A user who answered `backlog` to a prompt naming only X consented to X."""
    import sqlite3

    from opyt_core.paths import opyt_db
    conn = sqlite3.connect(opyt_db())
    try:
        conn.execute("CREATE TABLE atoms (atom_id TEXT)")
        conn.execute("INSERT INTO atoms VALUES ('substack:1')")
        conn.commit()
    finally:
        conn.close()

    from pipeline.kb import bookmark_catchup as bc
    assert bc.consented() is True                 # the older rail: implied by the same store
    assert ssc.consented() is False               # this one: the marker, and only the marker
    assert ssc.run_substack_saved_catchup()["status"] == "needs_consent"
    assert no_spend == []


def test_granting_consent_here_opts_into_nothing_else(rail_home, monkeypatch):
    """Two saved-content rails now answer to one `backlog` word, and they still keep separate
    markers: X's walk is a free cookie-scrape bounded by money, this one's is a Cloudflare-guarded
    reader endpoint bounded by throttling. Each reads its OWN marker before it spends, so a
    revoke or a rebuild of one never silently moves the other."""
    monkeypatch.delenv("OPYT_BOOKMARK_CATCHUP_CONSENT", raising=False)
    monkeypatch.delenv("OPYT_ORACLE_REFRESH_CONSENT", raising=False)
    from pipeline.kb import bookmark_catchup, oracle_refresh

    ssc.grant_consent()

    assert ssc._consent_marker() != bookmark_catchup._consent_marker()
    assert ssc._consent_marker() != oracle_refresh._consent_marker()
    assert oracle_refresh.consented() is False
    assert [p.name for p in sorted(rail_home.glob("*consent*"))] == [
        "substack_saved_catchup_consent"]


# ── the session gate ────────────────────────────────────────────────────────────
def test_a_hosted_home_with_no_substack_session_skips_instead_of_reporting_zero(
        rail_home, no_spend, monkeypatch):
    """Ungated, a hosted pass launches Chrome, is served the logged-out reader page and reports
    zero saved posts — which reads as "you saved nothing", not "Substack is not connected".

    Gated through `sources.substack.readable`, the ONE home for that rule, so this rail and
    `curation_catchup` cannot come to disagree about whether a session exists."""
    from pipeline.ingestion import hosted_browser, hosted_substack
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)
    monkeypatch.setattr(hosted_substack, "has_connection", lambda: False)
    # Override the fixture's default-present stub, and route the gate through the hosted mechanism
    # (`has_connection`) so this test still exercises the hosted answer, not a hardcoded False.
    monkeypatch.setattr(sub, "has_managed_substack_session", hosted_substack.has_connection)

    ssc.grant_consent()
    out = ssc.run_substack_saved_catchup()

    assert out["status"] == "no_session"
    assert no_spend == []


def test_a_local_home_with_no_managed_session_skips_instead_of_reporting_zero(
        rail_home, no_spend, monkeypatch):
    """The asymmetry is GONE as of 2026-09-13. The Substack reader uses only the OPYT-managed
    profile now — never the user's own browser — so with no managed session this rail skips on a
    local home exactly as the hosted test above does, rather than 401'ing and reporting "you
    saved nothing" where the truth is "Substack is not connected"."""
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: False)

    ssc.grant_consent()

    assert ssc.run_substack_saved_catchup()["status"] == "no_session"
    assert no_spend == []


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_catchup_skips_while_one_holds_the_lease(rail_home, no_spend):
    """Two passes at once double the request count against a Cloudflare-guarded endpoint that
    403s under bursty automation — the one failure this rail is most exposed to."""
    from pipeline.sync_lock import CatchupLock

    ssc.grant_consent()
    with CatchupLock("substack-saved-catchup") as held:
        assert held.acquired
        out = ssc.run_substack_saved_catchup()

    assert out["status"] == "already_running"
    assert no_spend == []


def test_its_lease_is_its_own_so_the_x_backlog_can_run_beside_it(rail_home, no_spend):
    """Both backlog rails are activated by the same `backlog` answer, so a user who connected
    both has them queued together. Sharing a lease name would make one silently skip the other
    and look, in the log, exactly like a pass with nothing to do."""
    from pipeline.sync_lock import CatchupLock

    ssc.grant_consent()
    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        assert ssc.run_substack_saved_catchup()["status"] == "ok"


# ── never raises ────────────────────────────────────────────────────────────────
def test_a_throwing_ingest_is_reported_not_propagated(rail_home, monkeypatch):
    """This runs in a detached child whose only caller is a `-m` entrypoint, so a propagated
    exception is a traceback in a log nobody opens — and it escapes before the `finally` that
    closes the connection."""
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())

    def boom(*a, **kw):
        raise RuntimeError("substack cloudflare challenge")

    monkeypatch.setattr("pipeline.kb.ingest_curation.sync_substack_saved", boom)
    ssc.grant_consent()
    out = ssc.run_substack_saved_catchup()
    assert out["status"] == "error"
    assert "cloudflare challenge" in out["error"]


def test_a_returned_error_is_not_reported_as_a_successful_run(rail_home, monkeypatch):
    """An adapter signals a hard stop by RETURNING `error`, never by raising. Passed through as
    `ok, added: 0`, a dead session is byte-identical to a week with nothing new saved — and a rail
    that believes it succeeded does not retry."""
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_curation.sync_substack_saved",
                        lambda *a, **kw: {"source": "substack-saved", "added": 0, "total": 0,
                                          "error": "SyncAuthError: cookie expired"})
    ssc.grant_consent()
    out = ssc.run_substack_saved_catchup()
    assert out["status"] == "error"
    assert out["added"] == 0


def test_a_blocked_run_is_blocked_not_error(rail_home, monkeypatch):
    """The distinction that must not collapse: Cloudflare lifts on its own, so a block is
    retryable; a dead session needs a person. `undetermined` is what the adapter already counts
    for a body it was STOPPED from fetching."""
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_curation.sync_substack_saved",
                        lambda *a, **kw: {"source": "substack-saved", "added": 3, "total": 3,
                                          "error": "cloudflare 403", "undetermined": 1})
    ssc.grant_consent()
    out = ssc.run_substack_saved_catchup()
    assert out["status"] == "blocked"
    assert out["added"] == 3


# ── the CLI ─────────────────────────────────────────────────────────────────────
def test_a_successful_run_exits_zero(rail_home, no_spend):
    """`sync_substack_saved` returns a run SUMMARY with no `status` key, so the wrapper stamps
    one on. Without it every wholly successful pass would exit 1 and the child would look
    permanently broken to anything reading exit codes."""
    ssc.grant_consent()
    assert ssc.main(["--once"]) == 0


def test_no_once_flag_prints_help_and_refuses(rail_home, no_spend, capsys):
    assert ssc.main([]) == 2
    assert no_spend == []
    capsys.readouterr()
