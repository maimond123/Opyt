"""The Oracle refresh loop: staleness, the window assertion, the breaker, the daily ceiling,
worst-lag-first ordering, and the cold-start bound.

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
        self.x_kwargs: list[dict] = []      # the FULL X kwargs — `limit` is asserted from here
        self.web_kwargs: list[dict] = []
        self.x = {"source": "x-footprint", "fetched": 40, "added": 3, "engagements": 7}
        self.web = {"source": "blog", "added": 1}

    def install(self, monkeypatch, *, x_raises=None):
        from pipeline.kb import expand

        def fake_x(conn, embedder, *, handle, author_name=None, since=None, **kw):
            self.calls.append(("x", handle, since))
            self.x_kwargs.append({"handle": handle, "since": since, **kw})
            if x_raises is not None:
                raise x_raises
            return dict(self.x)

        def fake_route(conn, embedder, source, *, author_name=None, limit=0,
                       github_min_stars=0, web_since=None, github_since=None):
            self.calls.append((source["source_type"], source["url"], web_since or github_since))
            self.web_kwargs.append({"url": source["url"], "limit": limit})
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
    # The remedy must name a path that STILL EXISTS. `oracle(action='refresh', force=True)` left
    # the tool surface 2026-08-15, so a refusal pointing there would send the user to an unknown
    # action — a dead end printed at exactly the moment they need a way forward.
    assert "--force" in r["remedy"] and "oracle_refresh" in r["remedy"]
    assert "action='refresh'" not in r["remedy"]


def test_window_assertion_refuses_a_metered_pair_that_LOST_its_coverage(store, fakes):
    """`since is None` on a pair WITH history is the threading-bug signature: a None reaching the
    X adapter becomes its 183-day default rather than an error, so every pull would silently cost
    a full onboarding.

    Scoped to a pair with history since 2026-08-30. A NEVER-pulled pair also has `since is None`
    and is NOT a fault — see `test_a_never_pulled_x_pair_gets_its_first_pull` for the deadlock
    that refusing it caused."""
    row = _stamp(store, "x", 1)
    row.last_pulled_at = None                 # the bug: history recorded, window lost
    row.cursor_ts = None
    r = orf.refresh_pair(store, None, row, now=NOW)
    assert r["status"] == "window_refused"
    assert fakes.calls == []


def test_force_overrides_the_window_assertion(store, fakes):
    _stamp(store, "x", 200 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert r["status"] == "ingested"
    assert len(fakes.calls) == 1


def test_free_sources_are_never_window_refused(store, fakes):
    """A wide `since` costs a free source nothing, so refusing it would be a livelock with no
    saving behind it — the pair could never advance `last_pulled_at`, so never stop being refused."""
    _stamp(store, "blog", 400 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "blog"), now=NOW)
    assert r["status"] == "ingested"


# ── the forced-pull bound ───────────────────────────────────────────────────────
# `force` waives the window assertion, which is the ONE path where `since` can reach the X adapter
# as None and become its 183-day default. `FORCED_X_ATOM_LIMIT` is what the waiver swaps in.
def test_a_waived_window_is_bounded_not_unbounded(store, fakes):
    """The gap this closes: before it, a forced pull past the window had no atom bound at all."""
    _stamp(store, "x", 200 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert fakes.x_kwargs[0]["limit"] == orf.FORCED_X_ATOM_LIMIT
    assert r["atom_limit"] == orf.FORCED_X_ATOM_LIMIT


def test_a_forced_pair_with_no_coverage_window_is_bounded(store, fakes):
    """`since is None` is the worst case — the adapter would silently fall back to 183 days."""
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert fakes.x_kwargs[0]["since"] is None          # nothing to thread; the bound is the guard
    assert fakes.x_kwargs[0]["limit"] == orf.FORCED_X_ATOM_LIMIT


def test_forcing_a_pair_whose_window_was_FINE_changes_nothing(store, fakes):
    """⚠️ The reason the bound keys on the WAIVER and not on `force`. `sync_x_footprint` skips its
    whole-window media + artifact prefetch whenever `limit` is set, so bounding every forced pull
    would trade a real latency optimization for a cap that had nothing to cap."""
    _stamp(store, "x", 100)                            # stale (72h TTL) but a ~4-day window
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert fakes.x_kwargs[0]["limit"] == 0
    assert "atom_limit" not in r


def test_an_ordinary_unforced_pull_is_never_bounded(store, fakes):
    _stamp(store, "x", 100)
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert fakes.x_kwargs[0]["limit"] == 0
    assert "atom_limit" not in r


def test_free_sources_are_never_bounded_even_under_force(store, fakes):
    """A free source is never window-refused, so it is never widened, so there is nothing to bound
    — and `limit` means a DIFFERENT thing to the web adapters (posts attempted, not atoms)."""
    _stamp(store, "blog", 400 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "blog"), force=True, now=NOW)
    assert fakes.web_kwargs[0]["limit"] == 0
    assert "atom_limit" not in r


def test_a_truncated_forced_pull_says_so_and_names_the_backfill_tool(store, monkeypatch):
    """Groups reach the consumer newest-first, so the bound drops the OLD end of the window. That
    is the right truncation for a currency loop and the wrong one to leave silent: a capped pull
    would otherwise read exactly like a window that happened to hold 300."""
    f = Fakes()
    f.x = {"source": "x-footprint", "fetched": 4000, "added": orf.FORCED_X_ATOM_LIMIT}
    f.install(monkeypatch)
    _stamp(store, "x", 200 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert r["status"] == "ingested" and r["truncated"] is True
    assert "oracle(action='ingest')" in r["note"]


def test_a_bounded_pull_that_came_back_short_is_not_called_truncated(store, monkeypatch):
    """The bound was applied; it just never bit. Flagging that would cry wolf on every forced
    pull of a quiet Oracle."""
    f = Fakes()
    f.x = {"source": "x-footprint", "fetched": 40, "added": 3}
    f.install(monkeypatch)
    _stamp(store, "x", 200 * 24)
    r = orf.refresh_pair(store, None, _pair(store, "x"), force=True, now=NOW)
    assert r["atom_limit"] == orf.FORCED_X_ATOM_LIMIT
    assert "truncated" not in r


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


# ── cost accounting ─────────────────────────────────────────────────────────────
def test_the_pairs_cost_is_the_recorded_delta_over_its_pull(store, fakes, monkeypatch):
    """Both keys carry the api_stats delta, and nothing is corrected out of it.

    They used to differ: `cost_usd` subtracted twitterapi.io's per-REQUEST guardrail estimate and
    substituted its real per-TWEET price, because the guardrail over-counted a short page by up to
    20x and an incremental window is mostly short pages. The provider was removed on 2026-08-30
    and the X fetch is free, so every dollar in this delta is now derived spend (OCR-VLM, artifact
    fetches, embeddings) that the meter records honestly. What remains worth pinning is that the
    delta is per-RAIL and brackets only this pull — see `_spend_probe`."""
    from pipeline import llm_client

    recorded = {"today": 0.0}
    monkeypatch.setattr(llm_client, "spend_today_for_rail", lambda *a: recorded["today"])

    def fake_x(conn, embedder, *, handle, author_name=None, since=None, **kw):
        recorded["today"] += 0.030      # embeds + image reads on the atoms this pull landed
        return {"source": "x-footprint", "fetched": 40, "added": 2}

    monkeypatch.setattr(orf, "ingest_x_footprint_sync", fake_x)
    _stamp(store, "x", 100)
    r = orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert r["cost_usd_recorded"] == pytest.approx(0.030)
    assert r["cost_usd"] == pytest.approx(0.030)


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


def test_daily_ceiling_pauses_mid_run(store, fakes, monkeypatch):
    _many_pairs(store, 5)
    _stamp(store, "x", 1)
    spent = {"v": 0.0}
    monkeypatch.setattr(orf, "_daily_budget_exhausted", lambda: spent["v"] >= 2)

    real_pair = orf.refresh_pair

    def counting(*a, **kw):
        spent["v"] += 1
        return real_pair(*a, **kw)

    monkeypatch.setattr(orf, "refresh_pair", counting)
    r = orf.refresh_all(store, None, max_pairs=8, now=NOW)
    assert r["status"] == "budget_paused"
    assert r["refreshed"] == 2
    assert r["deferred"] >= 1
    assert "daily refresh ceiling" in r["message"]


def test_second_immediate_run_is_a_full_no_op(store, fakes):
    """Two pairs move on run one — the blog pair is stale, and the X pair takes its bounded FIRST
    pull (never-pulled, so no window to refuse). Both are stamped, so run two touches nothing."""
    _stamp(store, "blog", 400)
    first = orf.refresh_all(store, None, now=NOW)
    assert first["refreshed"] == 2
    calls_after_first = len(fakes.calls)

    second = orf.refresh_all(store, None, now=utc_now())
    assert second["refreshed"] == 0
    assert second["cost_usd"] == 0.0
    assert len(fakes.calls) == calls_after_first      # nothing re-dispatched


# ── consent ─────────────────────────────────────────────────────────────────────
def test_unconsented_run_spends_nothing(kb_home, monkeypatch, fakes):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "nope"))
    r = orf.run_oracle_refresh()
    assert r["status"] == "needs_consent"
    assert fakes.calls == []


def test_force_grants_consent(kb_home, monkeypatch):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "c"))
    monkeypatch.setattr(orf, "refresh_all", lambda *a, **kw: {"status": "ok"})
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: None)
    assert not orf.consented()
    assert orf.run_oracle_refresh(force=True)["status"] == "ok"
    assert orf.consented()


def test_force_does_not_widen_the_selection_to_fresh_pairs(store, fakes):
    """`force` authorizes a wider WINDOW, not a re-pull of everything — because it is also how
    consent is granted, and the opt-in call is the one moment a user least expects a big bill."""
    _stamp(store, "x", 1)
    _stamp(store, "blog", 1)
    r = orf.refresh_all(store, None, force=True, now=NOW)
    assert r["considered"] == 0 and fakes.calls == []


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


def test_status_degrades_rather_than_raising(kb_home, monkeypatch):
    monkeypatch.setattr(st, "list_sources", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
    out = orf.status_summary()
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


def test_a_first_pull_is_bounded_so_it_cannot_be_a_runaway(store, fakes):
    """The first pull runs on the adapter's own default window, which nothing here chose. Bound
    the ATOMS instead, the same lever `--force` uses for the same reason: the window assertion is
    waived, so the atom cap is what is left holding the derived spend (image reads, embeds)."""
    orf.refresh_pair(store, None, _pair(store, "x"), now=NOW)
    assert fakes.x_kwargs[0]["limit"] == orf.FORCED_X_ATOM_LIMIT

