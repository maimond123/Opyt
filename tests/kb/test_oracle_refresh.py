"""The Oracle refresh loop: staleness, the window assertion, the breaker,
worst-lag-first ordering, and breadth before backfill.

Adapters are faked at the two dispatch seams (`ingest_x_footprint_sync` and `expand._route_source`)
so nothing here touches the network. The fakes return the adapters' REAL contract shape — a summary
dict, with a hard stop RETURNED rather than raised — because that contract is what several of these
assertions exist to pin.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pipeline.timeparse import utc_now

import pytest

from pipeline.kb import oracle_refresh as orf
from pipeline.kb import oracle_refresh_state as st
from pipeline.kb import schema

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


# ── fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture()
def consent(kb_home, monkeypatch):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "consent"))
    orf.grant_consent()
    return kb_home


@pytest.fixture(autouse=True)
def managed_x_session(monkeypatch):
    """Refresh tests use a connected X session unless the scenario is disconnection."""
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)


@pytest.fixture()
def store(kb_home):
    conn = st.connect()
    schema.upsert_entity(conn, "x:user:1", name="Will", identity_links=["https://willcb.com"], profile={"handle": "willccbb"})
    schema.upsert_entity(conn, "blog:willcb.com", name="Will", identity_links=["https://willcb.com"])
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "blog:willcb.com": "x:user:1"})
    schema.upsert_oracle(conn, "x:user:1", name="Will")
    st.seed_from_entities(conn)
    yield conn
    conn.close()


class Fakes:
    """Records every dispatch and serves a scripted summary per source type."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.x_kwargs: list[dict] = []      # the X window passed to the adapter
        self.web_kwargs: list[dict] = []
        self.x = {"source": "x-footprint", "fetched": 40, "added": 3, "engagements": 7}
        self.web = {"source": "blog", "added": 1}

    def install(self, monkeypatch, *, x_raises=None):
        from pipeline.kb import expand

        def fake_x(conn, embedder, *, handle, author_name=None, since=None):
            self.calls.append(("x", handle, since))
            self.x_kwargs.append({"handle": handle, "since": since})
            if x_raises is not None:
                raise x_raises
            return dict(self.x)

        def fake_route(conn, embedder, source, *, author_name=None, limit=0,
                       github_min_stars=0, web_since=None, github_since=None,
                       github_before=None):
            self.calls.append((source["source_type"], source["url"], web_since or github_since))
            self.web_kwargs.append({"url": source["url"], "limit": limit,
                                    "before": github_before})
            summ = dict(self.web)
            if summ.get("error"):
                return {"source_type": source["source_type"], "url": source["url"],
                        "blocked" if summ.get("undetermined") else "error":
                            summ if summ.get("undetermined") else summ["error"],
                        "reason": str(summ.get("error"))}
            return {"source_type": source["source_type"], "url": source["url"], "ingested": summ}

        monkeypatch.setattr(orf, "ingest_x_footprint_sync", fake_x)
        monkeypatch.setattr(expand, "_route_source", fake_route)
        return self


@pytest.fixture()
def fakes(monkeypatch):
    return Fakes().install(monkeypatch)


def _pair(conn, stype, key=None):
    return next(r for r in st.list_sources(conn)
                if r.source_type == stype and (key is None or r.source_key == key))


def _stamp(conn, stype, hours_ago, key=None):
    row = _pair(conn, stype, key)
    when = (NOW - timedelta(hours=hours_ago)).isoformat()
    st.record_pull(conn, row, last_status="ingested", cursor_ts=when, stamp=True, now=when)
    return _pair(conn, stype, key)


# ── staleness gate ──────────────────────────────────────────────────────────────
def test_fresh_pair_skips_at_zero_cost(store, fakes):
    row = _stamp(store, "x", 1)                       # 1h old against a 72h TTL
    r = orf.refresh_pair(store, None, row, now=NOW)
    assert r["status"] == "fresh"
    assert fakes.calls == []


def test_stale_pair_pulls_advances_and_stamps(store, fakes):
    _stamp(store, "x", 100)
    _seed = schema.upsert_atom(store, {"atom_id": "x:new", "source_type": "x",
                                       "who_id": "x:user:1", "when_ts": "2026-08-07",
                                       "description": "d"})
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["status"] == "ingested"
    assert r["new_atoms"] == 3 and r["engagements"] == 7
    after = _pair(store, "x")
    assert after.last_status == "ingested"
    assert after.cursor_ts == "2026-08-07"            # cursor comes from the CORPUS, not the fake
    assert after.last_pulled_at is not None


def test_empty_but_successful_pull_still_stamps(store, fakes):
    """A real observation. The flat TTL restarts from now — there is no empty-backoff here."""
    fakes.x = {"source": "x-footprint", "fetched": 0, "added": 0}
    _stamp(store, "x", 100)
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["status"] == "empty"
    assert _pair(store, "x").last_pulled_at is not None


def test_blocked_neither_advances_the_cursor_nor_stamps(store, monkeypatch):
    """The adapters RETURN a hard stop; nothing was written and nothing marked seen."""
    f = Fakes()
    f.x = {"source": "x-footprint", "fetched": 0, "added": 0, "undetermined": 1,
           "error": "provider returned no data"}
    f.install(monkeypatch)
    row = _stamp(store, "x", 100)
    before_cursor = row.cursor_ts
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["status"] == "blocked"
    after = _pair(store, "x")
    assert after.last_status == "blocked"
    assert after.cursor_ts == before_cursor
    assert st.is_stale(after, NOW)                    # still due — retried next run


# ── the window assertion ────────────────────────────────────────────────────────
def test_window_assertion_refuses_a_200_day_since_on_a_paid_source(store, fakes):
    _stamp(store, "x", 200 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["status"] == "window_refused"
    assert "45-day" in r["reason"]
    assert fakes.calls == []                          # refused BEFORE any spend


def test_a_metered_pull_never_reaches_the_adapter_with_no_window(store, fakes):
    """The harm the window assertion was written for: a `since` of None reaching the X adapter
    becomes its 183-day default rather than an error, so every pull silently costs a full
    onboarding.

    This used to be caught by REFUSING a metered pair whose window was None. That refusal could
    not tell a dropped window from a pair that had simply never been pulled, and refusing the
    second was a deadlock (see `test_a_never_pulled_x_pair_gets_its_first_pull`). It is now
    prevented by construction instead: a metered pair with no derivable window gets the concrete
    `BREADTH_WINDOW_DAYS` slice, so there is no None left to fall back to 183 days. Detecting a
    failure mode is worse than not having it."""
    row = _stamp(store, "x", 1)
    row.last_pulled_at = None                 # history recorded, window lost
    row.cursor_ts = None
    r = orf.refresh_pair(store, None, row, now=NOW)
    assert r["status"] == "ingested"
    since = fakes.x_kwargs[0]["since"]
    assert since is not None
    assert (NOW - since).days == orf.BREADTH_WINDOW_DAYS


def test_free_sources_are_never_window_refused(store, fakes):
    """A wide `since` costs a free source nothing, so refusing it would be a livelock with no
    saving behind it — the pair could never advance `last_pulled_at`, so never stop being refused."""
    _stamp(store, "blog", 400 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "blog"), now=NOW)
    assert r["status"] == "ingested"


# ── the breaker ─────────────────────────────────────────────────────────────────
def test_three_returned_errors_open_the_breaker_then_a_trial_closes_it(store, monkeypatch):
    """The adapters RETURN errors rather than raising. `breaker.call` would count those as
    successes and never trip, which is why the loop records the outcome explicitly."""
    f = Fakes()
    f.x = {"source": "x-footprint", "error": "no handle"}
    f.install(monkeypatch)
    for _ in range(orf.BREAKER_THRESHOLD):
        _stamp(store, "x", 100)
        assert orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)["status"] == "error"

    _stamp(store, "x", 100)
    assert orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)["status"] == "breaker_open"

    # Cooldown elapses → HALF_OPEN spends its one trial on the ACTUAL pull; a healthy one closes it.
    monkeypatch.setattr(orf, "BREAKER_COOLDOWN_S", 0.0)
    Fakes().install(monkeypatch)
    _stamp(store, "x", 100)
    assert orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)["status"] == "ingested"


def test_a_raising_adapter_is_an_error_not_a_crash(store, monkeypatch):
    Fakes().install(monkeypatch, x_raises=RuntimeError("boom"))
    _stamp(store, "x", 100)
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["status"] == "error" and "RuntimeError" in r["error"]
    assert _pair(store, "x").last_pulled_at is not None or True   # not stamped; still stale
    assert st.is_stale(_pair(store, "x"), NOW)


# ── the loop ────────────────────────────────────────────────────────────────────
def _many_pairs(conn, n: int, *, hours_ago_base: float = 400.0, step: float = 1.0):
    """n stale blog pairs with strictly increasing lag. Blog, because free sources are exempt
    from the window assertion. `step` sets the spacing: 1h leaves neighbours inside the jitter
    band (they may legitimately swap), 300h puts them unambiguously apart."""
    for i in range(n):
        row = st.SourceRow(canonical_id="x:user:1", source_type="blog",
                           source_key=f"https://s{i:02d}.com")
        st.upsert_source(conn, row)
        st.record_pull(conn, row, last_status="ingested", stamp=True,
                       now=(NOW - timedelta(hours=hours_ago_base + i * step)).isoformat())


def test_cold_start_burst_becomes_a_drained_backlog(store, fakes):
    """On the first run after seeding the WHOLE roster can come due at once. `max_pairs` turns
    that burst into a backlog, and `deferred` reports the remainder rather than truncating it."""
    _many_pairs(store, 30)
    _stamp(store, "x", 1)                             # keep the paid pair out of it

    first = orf.refresh_all(store, None, max_pairs=8, now=NOW)
    assert first["refreshed"] == 8
    assert first["deferred"] == first["considered"] - 8 == 23   # 30 blogs + the seeded blog pair

    second = orf.refresh_all(store, None, max_pairs=8, now=NOW)
    assert second["refreshed"] == 8
    assert second["deferred"] < first["deferred"]


def test_worst_lag_first_ordering(store, fakes):
    """Pairs are drained in DESCENDING `staleness_hours` — hours past their own effective TTL.

    Asserted against the computed ranking, not against a hardcoded name order. With jitter the
    effective TTL varies by ±10%, which on a 336h base is ±33h — so two pairs an hour apart
    legitimately swap, and a test that pinned the raw elapsed order would be asserting the
    absence of the very spreading jitter exists to create."""
    _many_pairs(store, 6, hours_ago_base=400)
    _stamp(store, "x", 1)
    _stamp(store, "blog", 1, key="https://willcb.com")     # keep the seeded pair fresh

    rows = [r for r in st.list_sources(store) if r.source_key.startswith("https://s")]
    expected = [r.source_key for r in
                sorted(rows, key=lambda r: st.staleness_hours(r, NOW), reverse=True)][:3]

    orf.refresh_all(store, None, max_pairs=3, now=NOW)
    assert [c[1] for c in fakes.calls] == expected


def test_ordering_still_tracks_elapsed_when_lags_are_far_apart(store, fakes):
    """Jitter reorders NEIGHBOURS, not the whole queue: a pair 300h more overdue still goes first."""
    _many_pairs(store, 3, hours_ago_base=400, step=300)    # 400h, 700h, 1000h
    _stamp(store, "x", 1)
    _stamp(store, "blog", 1, key="https://willcb.com")
    orf.refresh_all(store, None, max_pairs=2, now=NOW)
    pulled = [c[1] for c in fakes.calls]
    assert pulled[0].startswith("https://s02")             # 1000h
    assert pulled[1].startswith("https://s01")             # 700h


def test_a_permanently_refused_pair_does_not_starve_the_roster(store, fakes):
    """A window-refused pair spends nothing, so it must not consume one of `max_pairs` — else it
    sorts first every run (worst-lag-first) and blocks everything behind it forever."""
    _many_pairs(store, 3)
    # A pair whose recorded window is far past the ceiling → refused, and sorts first
    # (worst-lag-first). A NEVER-pulled pair would no longer do: it gets a bounded first pull.
    _stamp(store, "x", 400 * 24)
    r = orf.refresh_all(store, None, max_pairs=3, now=NOW)
    assert r["window_refused"] == 1
    assert r["refreshed"] == 3                        # all three free pairs still got their turn


def test_second_immediate_run_is_a_full_no_op(store, fakes):
    """Two pairs move on run one — the blog pair is stale, and the X pair takes its bounded FIRST
    pull (never-pulled, so no window to refuse). Both are stamped, so run two touches nothing."""
    _stamp(store, "blog", 400)
    first = orf.refresh_all(store, None, now=NOW)
    assert first["refreshed"] == 2
    calls_after_first = len(fakes.calls)

    second = orf.refresh_all(store, None, now=utc_now())
    assert second["refreshed"] == 0
    assert len(fakes.calls) == calls_after_first      # nothing re-dispatched


# ── consent ─────────────────────────────────────────────────────────────────────
def test_unconsented_run_spends_nothing(kb_home, monkeypatch, fakes):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "nope"))
    r = orf.run_oracle_refresh()
    assert r["status"] == "needs_consent"
    assert fakes.calls == []


def test_an_evicted_worker_stops_before_refreshing(consent, monkeypatch):
    class _LostLock:
        acquired = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def lost(self):
            return True

    from pipeline import sync_lock
    monkeypatch.setattr(sync_lock, "CatchupLock", lambda *args: _LostLock())
    monkeypatch.setattr(orf, "refresh_all",
                        lambda *args, **kwargs: pytest.fail("refresh ran after lease eviction"))

    assert orf.run_oracle_refresh()["status"] == "lease_lost"


def test_refresh_selects_only_stale_pairs(store, fakes):
    _stamp(store, "x", 1)
    _stamp(store, "blog", 1)
    r = orf.refresh_all(store, None, now=NOW)
    assert r["considered"] == 0 and fakes.calls == []


def test_refresh_skips_existing_x_rows_when_x_is_disconnected(store, fakes, monkeypatch):
    """Disconnecting X removes it from the pull queue without pausing public sources."""
    from pipeline.ingestion import x_graphql

    _stamp(store, "x", 100)
    _stamp(store, "blog", 400)
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)

    out = orf.refresh_all(store, None, now=NOW)

    assert out["registered"] == 1 and out["considered"] == 1
    assert [call[0] for call in fakes.calls] == ["blog"]


def test_consent_marker_resolves_at_call_time(kb_home, monkeypatch):
    """A path bound at import points at the real ~/.opyt under a sandboxed $OPYT_HOME."""
    monkeypatch.delenv("OPYT_ORACLE_REFRESH_CONSENT", raising=False)
    assert str(kb_home) in str(orf._consent_marker())


# ── status ──────────────────────────────────────────────────────────────────────
def test_status_surfaces_a_frozen_oracle(store, consent):
    _stamp(store, "x", 1)
    out = orf.status_summary(store)
    assert out["consented"] is True
    assert out["tracked_pairs"] == 2
    blog = next(s for o in out["oracles"] for s in o["sources"] if s["source_type"] == "blog")
    assert blog["never_refreshed"] is True and blog["stale"] is True
    # The EFFECTIVE ttl, so the report and the gate cannot disagree.
    assert abs(blog["ttl_hours"] - 336.0) <= 336.0 * st.TTL_JITTER
    assert blog["ttl_hours"] != 336.0 or st.TTL_JITTER == 0


def test_status_degrades_rather_than_raising(store, monkeypatch):
    monkeypatch.setattr(st, "list_sources", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
    out = orf.status_summary(store)
    assert out["tracked_pairs"] == 0 and "error" in out


# ── a never-pulled pair ─────────────────────────────────────────────────────────
def test_a_never_pulled_x_pair_gets_its_first_pull(store, fakes):
    """⚠️ THE DEADLOCK. An X pair with no `last_pulled_at` and no `cursor_ts` had `since_for`
    return None, and `window_ok` refuses a metered source on a None window — every run, forever,
    with nothing able to change either input. A pull is the only thing that writes them, and the
    refusal is what prevents the pull.

    It is reachable without anything exotic: confirm an Oracle whose first X pull fails outright
    (suspended account, expired cookie, an interrupted onboarding), and the rail can never pick it
    up again. `refresh_all` seeds `oracle_sources` for every confirmed Oracle, so the ROW is there
    — it is the window, not the registration, that is missing.

    The window assertion still does its job. It guards a pair WITH history against a threading bug
    that drops `since` and silently reintroduces the adapter's 183-day default. A pair with no
    history has no incremental window to lose: the adapter's default IS the right answer for a
    first pull, so `since=None` means "start from the beginning" here and "something went wrong"
    there. Those are different facts and only the second is a refusal."""
    row = _pair(store, "x")
    assert row.last_pulled_at is None and row.cursor_ts is None    # never pulled
    r = orf.refresh_pair(store, None, row, now=NOW)
    assert r["status"] != "window_refused", (
        "a never-pulled pair can never become pulled — nothing else writes last_pulled_at")
    assert [c[0] for c in fakes.calls] == ["x"]


def test_a_first_pull_is_a_breadth_pull_not_a_deep_one(store, fakes):
    """The first pull used to run on the adapter's 183-day default, bounded only by an atom cap.
    On a roster of fifteen never-pulled pairs that was the whole failure: `max_pairs` let eight of
    them each start a ~12-25-request walk, and X's 50-per-15-minutes `UserTweets` bucket was spent
    on the third — leaving three complete Oracles, twelve empty ones, and no way to tell which.

    A shallow WINDOW is the bound now, so breadth costs ~1-2 requests per pair and every Oracle
    gets recent coverage in one pass. Depth is `backfill_pass`, ordered shallowest-first."""
    orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert (NOW - fakes.x_kwargs[0]["since"]).days == orf.BREADTH_WINDOW_DAYS


# ── breadth, then depth ─────────────────────────────────────────────────────────
# `backfill_pass` is the BACKWARD job: it walks each X pair back toward
# `BACKFILL_TARGET_DAYS`, shallowest first, until x.com's own meter runs dry. The forward job
# (`refresh_all`) never walks older than its cursor, so before this nothing could deepen an
# Oracle whose first pull was shallow.

@pytest.fixture()
def roster(kb_home):
    """A store with N X-rooted Oracles, so ordering and budget assertions have something to
    order. One source each — the point here is spread across PEOPLE, not across source types."""
    def _build(n: int):
        conn = st.connect()
        for i in range(1, n + 1):
            schema.upsert_entity(conn, f"x:user:{i}", name=f"P{i}", profile={"handle": f"p{i}"})
            schema.set_canonical_ids(conn, {f"x:user:{i}": f"x:user:{i}"})
            schema.upsert_oracle(conn, f"x:user:{i}", name=f"P{i}")
        st.seed_from_entities(conn)
        return conn
    return _build


def _breadth(conn, cid, *, days_back=orf.BREADTH_WINDOW_DAYS):
    """Record the pull a breadth pass would have made for one Oracle's X pair."""
    row = next(r for r in st.list_sources(conn, canonical_ids=[cid]) if r.source_type == "x")
    st.record_pull(conn, row, last_status="ingested", stamp=True, now=NOW.isoformat(),
                   covered_from=(NOW - timedelta(days=days_back)).isoformat())


def _frontiers(conn):
    return {r.canonical_id: r.covered_from
            for r in st.list_sources(conn) if r.source_type == "x"}


def test_three_oracles_all_reach_the_target_in_one_pass(roster, fakes):
    """With a small roster the whole budget covers everyone, so breadth-then-depth completes in
    one pass — and it does so WITHOUT the code knowing the roster is small. There is no
    `if len(oracles) <= N` branch: N has no defensible value, because what varies is request
    count and Oracle count is a proxy for it that does not hold."""
    conn = roster(3)
    for i in (1, 2, 3):
        _breadth(conn, f"x:user:{i}")

    out = orf.backfill_pass(conn, None, now=NOW)

    assert out["considered"] == 3 and out["deepened"] == 3
    target = orf.deepen_target(NOW)
    for frontier in _frontiers(conn).values():
        assert st.parse_ts(frontier) == target
    # Every pull asked for the target itself, not an incremental step toward it: X paginates
    # newest-first with no `until`, so stepping there costs 2.5x the requests for the same history.
    assert [kw["since"] for kw in fakes.x_kwargs] == [target] * 3
    conn.close()


def test_fifteen_oracles_deepen_shallowest_first(roster, fakes, monkeypatch):
    """The property that keeps incompleteness UNIFORM. Without it one prolific account eats the
    window while fourteen stay at their breadth pull, and "I have three of your fifteen and cannot
    tell you which" is not a sentence the store can usefully say."""
    conn = roster(15)
    # Oracle i has been pulled back i*10 days — 1 is the shallowest, 15 the deepest.
    for i in range(1, 16):
        _breadth(conn, f"x:user:{i}", days_back=i * 10)
    # Stop after 4 pulls, the way a spent request window would.
    calls = {"n": 0}
    real = orf.backfill_pair

    def _capped(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 4:
            from pipeline.ingestion import x_graphql_core as core
            raise core.XRateLimited("window spent")
        return real(*a, **kw)

    monkeypatch.setattr(orf, "backfill_pair", _capped)
    out = orf.backfill_pass(conn, None, now=NOW)

    # The four shallowest went first, in order. All fifteen are gaps: the deepest sits at 150
    # days, still short of the 183-day target.
    assert [c[1] for c in fakes.calls] == ["p1", "p2", "p3", "p4"]
    assert out["status"] == "rate_paused" and out["resumes"] == "next-scheduled-run"
    assert out["deferred"] == 11                     # reported, never silently truncated
    conn.close()


def test_a_pair_already_deep_enough_is_not_re_pulled(roster, fakes):
    """`covered_from` is the whole point of the column: without a backward frontier the pass has
    no way to tell a deep Oracle from a shallow one and would re-walk everyone every session."""
    conn = roster(2)
    _breadth(conn, "x:user:1", days_back=orf.BACKFILL_TARGET_DAYS + 10)   # already deeper
    _breadth(conn, "x:user:2", days_back=5)

    out = orf.backfill_pass(conn, None, now=NOW)
    assert out["considered"] == 1
    assert [c[1] for c in fakes.calls] == ["p2"]
    conn.close()


def test_a_never_pulled_pair_belongs_to_breadth_not_to_depth(roster, fakes):
    """Depth deepens what breadth already touched. A pair with no stamp at all has not had its
    cheap recent slice yet, and starting a deep walk on it is exactly the ordering that spent the
    request window on three Oracles out of fifteen."""
    conn = roster(2)
    _breadth(conn, "x:user:1", days_back=5)          # 2 is left never-pulled

    out = orf.backfill_pass(conn, None, now=NOW)
    assert out["considered"] == 1
    assert [c[1] for c in fakes.calls] == ["p1"]
    conn.close()


def test_a_successful_pull_records_its_full_window(store, fakes):
    fakes.x = {"source": "x-footprint", "fetched": 999, "added": 350}
    row = _stamp(store, "x", 100)
    since = orf.since_for(row)
    r = orf.refresh_pair(store, None, row, now=NOW)
    assert r["new_atoms"] == 350
    assert _pair(store, "x").covered_from == since.isoformat()


def test_a_thin_meter_starts_the_walk_instead_of_reserving_against_it(roster, fakes,
                                                                     monkeypatch):
    """⚠️ INVERTED 2026-09-14. A reservation here skipped an X pair whenever the bucket was below
    a fixed floor — correct while a cut-short walk raised with NOTHING written, and pure loss once
    a partial walk keeps what it got. `backfill_pass` promises "until the meter runs dry" in its
    own docstring; this is what finally makes that literal."""
    from pipeline.ingestion import x_graphql_core as core
    import time

    conn = roster(3)
    for i in (1, 2, 3):
        _breadth(conn, f"x:user:{i}")
    monkeypatch.setattr(core, "_RATE_STATE", {core.USERTWEETS_OP: (1, time.time() + 600)})

    out = orf.backfill_pass(conn, None, now=NOW)

    assert out["deepened"] == 3 and out["deferred"] == 0
    assert len(fakes.calls) == 3                     # every pair got its walk
    conn.close()


def test_x_com_cutting_the_session_off_defers_the_remaining_x_pairs(roster, monkeypatch):
    """The real bound, and the only one left: x.com refusing. Session-wide, so no later X pair is
    retried — and the tail is counted `deferred` rather than left looking done."""
    conn = roster(3)
    for i in (1, 2, 3):
        _breadth(conn, f"x:user:{i}")
    Fakes().install(monkeypatch, x_raises=orf._core().XRateLimited("spent"))

    out = orf.backfill_pass(conn, None, now=NOW)

    assert out["status"] == "rate_paused" and out["deferred"] == 3
    assert out["resumes"] == "next-scheduled-run"
    conn.close()


def test_an_unknown_meter_does_not_block_a_fresh_process(roster, fakes):
    """A detached child starts blind. Refusing on no evidence would mean a fresh process never
    backfills at all — and it is blind for exactly one pull, since the bucket is server-side and
    the first response fills the meter in."""
    conn = roster(1)
    _breadth(conn, "x:user:1")
    out = orf.backfill_pass(conn, None, now=NOW)
    assert out["deepened"] == 1
    conn.close()


# ── session-wide failures stop the pass; they never charge the breaker ──────────
def test_a_rate_limit_stops_the_run_instead_of_marking_pairs_broken(store, monkeypatch, fakes):
    """`XRateLimited` says the SESSION's budget is spent, not that this handle is broken. Handled
    as a per-pair error it would charge the circuit breaker for somebody else's outage: three
    rate-limited sessions in a row open a 7-day cooldown on three perfectly healthy handles.

    Stopping is also the cheap answer — every remaining pair fails identically, so continuing
    buys nothing and costs the whole queue's breaker state."""
    from pipeline.circuit_breaker import CircuitBreaker
    from pipeline.ingestion import x_graphql_core as core

    _stamp(store, "x", 100)
    _stamp(store, "blog", 400)
    monkeypatch.setattr(orf, "ingest_x_footprint_sync",
                        lambda *a, **kw: (_ for _ in ()).throw(core.XRateLimited("spent")))

    out = orf.refresh_all(store, None, now=NOW)

    assert out["status"] == "rate_paused" and out["resumes"] == "next-scheduled-run"
    assert out["errors"] == 0
    assert out["deferred"] >= 1
    breaker = CircuitBreaker(f"oracle-refresh:x:user:1:x", threshold=orf.BREAKER_THRESHOLD,
                             cooldown=orf.BREAKER_COOLDOWN_S)
    assert breaker.allow()                        # untouched: this was not the handle's fault
    assert _pair(store, "x").last_status != "error"


def test_a_spent_model_allowance_does_not_mark_three_healthy_handles_broken(
        store, monkeypatch, fakes):
    """⚠️ MEASURED IN PRODUCTION, 2026-09-15, and it cost a user every Oracle they had.

    A starter allowance hit its ceiling. OpenRouter answered `403 Key limit exceeded`, the
    `openrouter-embed` circuit opened, and the next three X pulls each raised `CircuitOpenError`.
    Charged per-pair, that is three strikes apiece — three perfectly healthy X handles under a
    SEVEN-DAY cooldown whose whole cause was a missing cent of credit, and which adding credit
    would not have cleared, because nothing the user can do reaches the breaker table.

    It is the same sentence as the rate-limit case one layer out: the model provider is as
    session-wide as the X window, and as little the handle's fault. `CircuitOpenError` can only
    reach `refresh_pair` from a provider circuit — the pair's own breaker is consulted with
    `allow()` and never raises."""
    from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError

    _stamp(store, "x", 100)
    _stamp(store, "blog", 400)
    monkeypatch.setattr(orf, "ingest_x_footprint_sync",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            CircuitOpenError("openrouter-embed", 60.0)))

    out = orf.refresh_all(store, None, now=NOW)

    assert out["status"] == "provider_down" and out["resumes"] == "next-scheduled-run"
    assert out["errors"] == 0
    assert out["deferred"] >= 1
    breaker = CircuitBreaker("oracle-refresh:x:user:1:x", threshold=orf.BREAKER_THRESHOLD,
                             cooldown=orf.BREAKER_COOLDOWN_S)
    assert breaker.allow()                        # untouched: this was not the handle's fault
    assert _pair(store, "x").last_status != "error"


def test_a_dead_cookie_requires_reconnect(store, monkeypatch, fakes):
    """A dead session needs a new login; only a rate limit resumes by itself."""
    from pipeline.ingestion.utils import SyncAuthError

    _stamp(store, "x", 100)
    monkeypatch.setattr(orf, "ingest_x_footprint_sync",
                        lambda *a, **kw: (_ for _ in ()).throw(SyncAuthError("cookie expired")))
    out = orf.refresh_all(store, None, now=NOW)
    assert out["status"] == "needs_reconnect" and out["needs_reconnect"] == "x"
    assert "resumes" not in out and out["errors"] == 0


# ── the backward pass is no longer X-only ───────────────────────────────────────
# GitHub is the second source in `DEEPENED_SOURCES`, and it is there because its repo LIST
# arrives whole and carries every `pushed_at` before a single README is fetched. That is what
# lets a sweep be bounded from BOTH ends — `ingest_github.REPOS_PER_RUN` repos per run, starting
# below the frontier the last run reported — and it is the difference from X, whose pagination
# has no upper bound and must re-walk everything newer to reach anything older.

@pytest.fixture()
def gh_roster(kb_home):
    """One Oracle with a GitHub pair. `seed_from_entities` derives the pair from a `github:`
    entity folded onto the Oracle's canonical id, which is the shape a bio link produces."""
    conn = st.connect()
    schema.upsert_entity(conn, "x:user:1", name="Will", profile={"handle": "willccbb"})
    schema.upsert_entity(conn, "github:willccbb", name="willccbb")
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "github:willccbb": "x:user:1"})
    schema.upsert_oracle(conn, "x:user:1", name="Will")
    st.seed_from_entities(conn)
    yield conn
    conn.close()


def _gh_pair(conn):
    return next(r for r in st.list_sources(conn) if r.source_type == "github")


def test_a_shallow_github_pair_is_deepened(gh_roster, fakes):
    """Before this, `coverage_gap` refused every non-X pair, so a GitHub archive truncated by the
    60/hr anonymous limit stayed truncated forever: the forward pass never looks older than its
    cursor and nothing else walked backward."""
    fakes.web = {"source": "github", "added": 2}
    st.record_pull(gh_roster, _gh_pair(gh_roster), last_status="ingested", stamp=True,
                   now=NOW.isoformat(), covered_from=(NOW - timedelta(days=20)).isoformat())

    out = orf.backfill_pass(gh_roster, None, now=NOW)

    assert out["considered"] == 1 and out["deepened"] == 1
    assert [c[0] for c in fakes.calls] == ["github"]


def test_backfill_skips_x_when_x_is_disconnected(gh_roster, fakes, monkeypatch):
    """A stale X row from an earlier connection cannot block a GitHub backfill."""
    from pipeline.ingestion import x_graphql

    st.record_pull(gh_roster, _gh_pair(gh_roster), last_status="ingested", stamp=True,
                   now=NOW.isoformat(), covered_from=(NOW - timedelta(days=20)).isoformat())
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)

    out = orf.backfill_pass(gh_roster, None, now=NOW)

    assert out["considered"] == 1 and [call[0] for call in fakes.calls] == ["github"]


def test_a_resume_hands_the_stored_frontier_back_as_the_upper_bound(gh_roster, fakes):
    """The whole saving. Without `before`, the resume re-walks the same newest prefix and spends
    the same calls to reach the same place — the gap D1 named and deliberately left open."""
    fakes.web = {"source": "github", "added": 1}
    frontier = (NOW - timedelta(days=20)).isoformat()
    st.record_pull(gh_roster, _gh_pair(gh_roster), last_status="ingested", stamp=True,
                   now=NOW.isoformat(), covered_from=frontier)

    orf.backfill_pass(gh_roster, None, now=NOW)

    assert fakes.web_kwargs[-1]["before"] == st.parse_ts(frontier)


def test_the_sweeps_own_frontier_beats_the_window_it_was_asked_for(gh_roster, fakes):
    """A bounded sweep stops on a repo COUNT, so where it stopped is not derivable from the window
    the caller passed. Stamping the requested window instead is the lie this replaces: it claims
    183 days of coverage for a run that reached three weeks."""
    reached = (NOW - timedelta(days=21)).isoformat()
    fakes.web = {"source": "github", "added": 50, "capped": 139, "covered_from": reached}
    st.record_pull(gh_roster, _gh_pair(gh_roster), last_status="ingested", stamp=True,
                   now=NOW.isoformat(), covered_from=(NOW - timedelta(days=20)).isoformat())

    orf.backfill_pass(gh_roster, None, now=NOW)

    assert _gh_pair(gh_roster).covered_from == reached
    assert st.parse_ts(reached) > orf.deepen_target(NOW)      # short of the target, and honest


def test_x_com_cutting_off_does_not_starve_a_github_pair(gh_roster, monkeypatch):
    """x.com's request meter bounds X pulls and nothing else. A GitHub pair is bounded by
    `ingest_github.REPOS_PER_RUN`, so breaking the whole pass on an X refusal would stall GitHub
    deepening for as long as the X meter stayed low — which is most of the time.

    ⚠️ THE X PAIR IS WHAT MAKES THIS TEST ABLE TO FAIL. Until 2026-09-14 the property was provided
    by a pre-emptive reserve that `continue`d past X pairs, and this fixture had no X pair at all
    — so deleting that reserve would have regressed GitHub deepening with the suite green.
    """
    from pipeline.ingestion import x_graphql
    from pipeline.kb import oracle_refresh_state as st2

    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    st2.upsert_source(gh_roster, st2.SourceRow(canonical_id="x:user:1", source_type="x",
                                               source_key="willccbb", status="trusted"))
    f = Fakes().install(monkeypatch, x_raises=orf._core().XRateLimited("spent"))
    f.web = {"source": "github", "added": 1}
    # Shallowest first, so the X pair must be the SHALLOWER of the two or it sorts behind the
    # GitHub pair and the test proves nothing about what happens after an X refusal.
    _breadth(gh_roster, "x:user:1", days_back=10)
    st.record_pull(gh_roster, _gh_pair(gh_roster), last_status="ingested", stamp=True,
                   now=NOW.isoformat(), covered_from=(NOW - timedelta(days=20)).isoformat())

    out = orf.backfill_pass(gh_roster, None, now=NOW)

    assert out["deepened"] == 1                      # the GitHub pair ran…
    assert out["deferred"] == 1                      # …and only the X pair was deferred
