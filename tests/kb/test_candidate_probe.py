"""The light timeline pull over the Oracle candidate list.

Offline throughout — `fetch_user_tweets` is monkeypatched, so these pin the LOOP's decisions, not
the live scrape. The decisions worth pinning are the ones that cost the scarce thing (~169 shared
requests/hour) or that silently mislabel a person:

  • who is due, and who is excluded from the queue at all
  • the four outcome states, and which of them buy a candidate a TTL's worth of silence
  • a rate limit STOPS the run; a per-candidate error skips only that candidate
  • the curation filter runs and the substance filter does not (short posts must survive)
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql_core as core
from pipeline.kb import candidate_probe as cp
from pipeline.kb import probe_store, resolve, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _candidate(conn, uid: str, *, signals=(("follow", "x"),), name=None, handle=None):
    eid = f"x:user:{uid}"
    schema.upsert_entity(conn, eid, name=name or f"P{uid}", profile={"handle": handle or f"p{uid}"})
    for st, pf in signals:
        schema.add_signal(conn, eid, st, pf)
    return eid


def _tweet(tid: str, *, uid: str = "11", user: str = "p11", text: str = "on protein folding",
           reply_to: str | None = None, conv: str | None = None) -> dict:
    """A NORMALIZED tweet — the shape `fetch_user_tweets` returns."""
    t = {"id": tid, "text": text, "createdAt": "Mon Aug 11 10:00:00 +0000 2026",
         "author": {"userName": user, "name": user, "id": uid, "site": ""},
         "entities": {}, "extendedEntities": {}, "likeCount": 1, "replyCount": 0,
         "url": f"https://x.com/{user}/status/{tid}",
         "conversationId": conv or tid, "isQuote": False, "isRetweet": False,
         "isReply": bool(reply_to), "inReplyToUserId": reply_to or ""}
    return t


def _serve(monkeypatch, by_user: dict):
    """Stub the fetch. A value may be a tweet list, or an Exception to raise. Records the order
    accounts were requested in — that IS the queue order under a bounded budget."""
    asked: list[str] = []

    def _fake(cookies, headers, user_id, *, pages=1, page_size=20, after_page=None):
        asked.append(str(user_id))
        v = by_user.get(str(user_id), [])
        if isinstance(v, Exception):
            raise v
        # Mimic the real walk's ONE call per landed page, so a test that asserts on the caller's
        # stop/pace hook exercises the same contract the primitive offers.
        if after_page is not None and v:
            after_page(v)
        return v

    monkeypatch.setattr(core, "fetch_user_tweets", _fake)
    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"auth_token": "t"})
    monkeypatch.setattr(core, "auth_headers", lambda cookies, referer: {})
    return asked


def _run(conn, embedder, monkeypatch, by_user, **kw):
    _serve(monkeypatch, by_user)
    kw.setdefault("pace_seconds", 0)          # no real sleeping in tests
    return cp.probe_candidates(conn, embedder, **kw)


# ── the queue ─────────────────────────────────────────────────────────────────

def test_queue_is_in_screen_rank_order(conn):
    _candidate(conn, "1", signals=(("like", "x"),))                    # 1 distinct
    _candidate(conn, "2", signals=(("save", "x"), ("follow", "x")))    # 2 distinct → first
    assert [c["who_id"] for c in cp.candidate_queue(conn)] == ["x:user:2", "x:user:1"]


def test_queue_honors_min_signals(conn):
    _candidate(conn, "1", signals=(("like", "x"),))
    _candidate(conn, "2", signals=(("save", "x"), ("follow", "x")))
    assert [c["who_id"] for c in cp.candidate_queue(conn, min_signals=2)] == ["x:user:2"]


def test_queue_excludes_confirmed_oracles(conn):
    """Their real footprint is already in `atoms` — probing them would duplicate TRUSTED content
    into the untrusted store."""
    _candidate(conn, "1")
    schema.upsert_oracle(conn, "x:user:1", name="P1", source="screen")
    assert cp.candidate_queue(conn) == []


def test_queue_excludes_candidates_with_no_x_identity(conn):
    """A Substack-only subscription has no timeline on this path — absent, not failed."""
    schema.upsert_entity(conn, "substack:carol", name="Carol")
    schema.add_signal(conn, "substack:carol", "subscribe", "substack")
    resolve.resolve_entities(conn)
    assert cp.candidate_queue(conn) == []


def test_queue_skips_a_fresh_snapshot_and_re_admits_a_stale_one(conn):
    _candidate(conn, "1")
    probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_OK, atoms=3)
    assert cp.candidate_queue(conn, ttl_days=30) == []
    assert [c["who_id"] for c in cp.candidate_queue(conn, ttl_days=0)] == ["x:user:1"]


@pytest.mark.parametrize("status", [probe_store.STATUS_EMPTY, probe_store.STATUS_UNAVAILABLE])
def test_a_real_observation_buys_silence(conn, status):
    """`empty` and `unavailable` are FACTS about the candidate, so they wait out the TTL like `ok`.
    Re-fetching them every run is exactly the wasted request budget the state table exists for."""
    _candidate(conn, "1")
    probe_store.record_pull(conn, "x:user:1", status)
    assert cp.candidate_queue(conn, ttl_days=30) == []


def test_a_failure_is_always_due_again(conn):
    """The fail-safe invariant: a failed external call records what happened and never marks
    unfinished work done — so it must NOT buy the candidate a TTL's worth of silence."""
    _candidate(conn, "1")
    probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_FAILED, detail="boom")
    assert [c["who_id"] for c in cp.candidate_queue(conn, ttl_days=30)] == ["x:user:1"]


# ── the four outcomes ─────────────────────────────────────────────────────────

def test_content_lands_in_the_probe_store_only(conn, fake_embedder, monkeypatch):
    _candidate(conn, "11", handle="p11")
    out = _run(conn, fake_embedder, monkeypatch,
               {"11": [_tweet("1", text="an agent framework for autonomous tools")]})
    assert out["by_status"][probe_store.STATUS_OK] == 1
    assert out["atoms"] == 1
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT atom_id FROM probe_atoms").fetchone()[0] == "xprobe:1"


def test_zero_tweets_is_recorded_as_empty_not_as_a_failure(conn, fake_embedder, monkeypatch):
    _candidate(conn, "11")
    out = _run(conn, fake_embedder, monkeypatch, {"11": []})
    assert out["by_status"][probe_store.STATUS_EMPTY] == 1
    assert probe_store.pull_states(conn)["x:user:11"]["status"] == probe_store.STATUS_EMPTY


def test_an_unavailable_account_is_its_own_state(conn, fake_embedder, monkeypatch):
    _candidate(conn, "11")
    out = _run(conn, fake_embedder, monkeypatch,
               {"11": core.XUserUnavailable("x:user:11 timeline unreadable: Suspended")})
    assert out["by_status"][probe_store.STATUS_UNAVAILABLE] == 1
    assert "Suspended" in probe_store.pull_states(conn)["x:user:11"]["detail"]


def test_one_candidates_failure_does_not_stop_the_others(conn, fake_embedder, monkeypatch):
    _candidate(conn, "11", signals=(("save", "x"), ("follow", "x")))     # ranks first
    _candidate(conn, "22", signals=(("follow", "x"),))
    out = _run(conn, fake_embedder, monkeypatch,
               {"11": RuntimeError("transient"), "22": [_tweet("9", uid="22", user="p22")]})
    assert out["by_status"] == {probe_store.STATUS_OK: 1, probe_store.STATUS_EMPTY: 0,
                                probe_store.STATUS_UNAVAILABLE: 0, probe_store.STATUS_FAILED: 1}
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 0     # SKIP: no partial write
    assert probe_store.count_probe_atoms(conn, "x:user:22") == 1


# ── session-wide stops ────────────────────────────────────────────────────────

def test_a_rate_limit_stops_the_run_instead_of_burning_the_queue(conn, fake_embedder, monkeypatch):
    """Every remaining request would 429 identically. Marking the rest `failed` would be a lie
    about them AND would hammer x.com on the way to telling it."""
    for i in (1, 2, 3):
        _candidate(conn, str(i))
    out = _run(conn, fake_embedder, monkeypatch,
               {"1": core.XRateLimited("429"), "2": [], "3": []})
    assert out["stopped"] == "rate_limited"
    assert out["requests"] == 0
    assert probe_store.pull_states(conn) == {}       # nothing observed → nothing recorded
    assert len(cp.candidate_queue(conn)) == 3        # the whole queue is still due


def test_a_dead_session_stops_the_run(conn, fake_embedder, monkeypatch):
    from pipeline.ingestion.utils import SyncAuthError
    _candidate(conn, "1")
    _candidate(conn, "2")
    out = _run(conn, fake_embedder, monkeypatch, {"1": SyncAuthError("cookie expired"), "2": []})
    assert out["stopped"] == "auth"
    assert probe_store.pull_states(conn) == {}


def test_no_x_session_degrades_rather_than_crashing(conn, fake_embedder, monkeypatch):
    from pipeline.ingestion.utils import SyncAuthError
    _candidate(conn, "1")
    monkeypatch.setattr(core, "read_x_cookies",
                        lambda profile=None: (_ for _ in ()).throw(SyncAuthError("no browser")))
    out = cp.probe_candidates(conn, fake_embedder, pace_seconds=0)
    assert out["stopped"] == "auth" and out["queued"] == 1


# ── the budget ────────────────────────────────────────────────────────────────

def test_max_candidates_is_the_request_budget(conn, fake_embedder, monkeypatch):
    for i in (1, 2, 3):
        _candidate(conn, str(i))
    asked = _serve(monkeypatch, {str(i): [] for i in (1, 2, 3)})
    out = cp.probe_candidates(conn, fake_embedder, max_candidates=2, pace_seconds=0)
    assert len(asked) == 2 and out["requests"] == 2
    assert out["remaining"] == 1                     # resumable — the third is still due


# ── filtering: the curation filter runs, the substance filter does not ────────

def test_self_thread_becomes_one_atom_and_replies_to_others_are_dropped(
        conn, fake_embedder, monkeypatch):
    _candidate(conn, "11", handle="p11")
    _run(conn, fake_embedder, monkeypatch, {"11": [
        _tweet("1", conv="1", text="thread root about protein folding"),
        _tweet("2", conv="1", reply_to="11", text="continuation of my own thread"),
        _tweet("3", reply_to="99", text="replying to a stranger"),
    ]})
    rows = [r[0] for r in conn.execute("SELECT atom_id FROM probe_atoms ORDER BY atom_id")]
    assert rows == ["xprobe:1"]                      # ONE atom: root + self-reply, stranger gone


def test_a_one_line_aphorism_is_kept(conn, fake_embedder, monkeypatch):
    """The substance filter's 200-char bar must NOT run here. A Proposer asks what field someone is
    in, and a short post answers that — dropping it reintroduces the essayist bias."""
    _candidate(conn, "11", handle="p11")
    _run(conn, fake_embedder, monkeypatch, {"11": [_tweet("1", text="AI is biology now.")]})
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1


# ── idempotency ───────────────────────────────────────────────────────────────

# ── a write shortfall is NOT a successful observation ─────────────────────────
#
# CAUGHT LIVE 2026-08-11. The first real run had no OPENROUTER_API_KEY at the sandboxed
# $OPYT_HOME: 14 snapshots rendered, 0 atoms stored, and `probe_pulls` recorded `ok, atoms=0` —
# buying that candidate 30 days of silence for content that does not exist. The sink swallows a
# failed embed per atom BY DESIGN (so one poison chunk cannot sink a batch), so a systemic failure
# is invisible to the caller except as a shortfall. These two tests are that shortfall.

def test_a_total_embed_failure_records_failed_not_ok(conn, monkeypatch):
    from tests.kb.conftest import RecordingEmbedder

    _candidate(conn, "11", handle="p11")
    # Poisons the CONTENT but not the preflight text — so the run gets past the preflight and
    # dies exactly where the live failure did: inside the sink, silently, per atom.
    dead = RecordingEmbedder(poison="agent")
    _serve(monkeypatch, {"11": [_tweet("1", text="an agent framework")]})
    monkeypatch.setattr(cp, "assert_model", lambda *a, **k: None)
    out = cp.probe_candidates(conn, dead, pace_seconds=0)

    assert out["by_status"][probe_store.STATUS_FAILED] == 1
    assert out["by_status"][probe_store.STATUS_OK] == 0
    assert probe_store.count_probe_atoms(conn) == 0
    # ...and therefore still due, rather than frozen behind the TTL.
    assert [c["who_id"] for c in cp.candidate_queue(conn, ttl_days=30)] == ["x:user:11"]


def test_a_partial_write_also_records_failed(conn, monkeypatch):
    """Half-stored is not stored. Retrying costs one request and re-embeds only the gap (the rest
    hash-skip), so freezing a hole is the expensive option, not the cheap one."""
    from tests.kb.conftest import RecordingEmbedder

    _candidate(conn, "11", handle="p11")
    picky = RecordingEmbedder(poison="rollup")     # only the second post's batch dies
    _serve(monkeypatch, {"11": [_tweet("1", text="an agent framework"),
                                _tweet("2", text="a crypto rollup proof")]})
    monkeypatch.setattr(cp, "assert_model", lambda *a, **k: None)
    out = cp.probe_candidates(conn, picky, pace_seconds=0)

    assert out["by_status"][probe_store.STATUS_FAILED] == 1
    assert 0 < probe_store.count_probe_atoms(conn) < 2      # some landed, not all
    assert "atoms stored" in probe_store.pull_states(conn)["x:user:11"]["detail"]


def test_an_unusable_embedder_spends_no_x_requests(conn, monkeypatch):
    """The preflight. `assert_model` checks embedding IDENTITY and passes happily with no API key,
    so without this the failure surfaces 25 requests later having stored nothing."""
    from tests.kb.conftest import RecordingEmbedder

    _candidate(conn, "11")
    _candidate(conn, "22")
    asked = _serve(monkeypatch, {"11": [], "22": []})
    monkeypatch.setattr(cp, "assert_model", lambda *a, **k: None)
    out = cp.probe_candidates(conn, RecordingEmbedder(poison="preflight"), pace_seconds=0)

    assert out["stopped"] == "embedder"
    assert asked == []                                     # zero requests of the shared budget
    assert probe_store.pull_states(conn) == {}             # nothing observed → nothing recorded


def test_a_filtered_out_page_is_still_a_successful_observation(conn, fake_embedder, monkeypatch):
    """`submitted == 0` is NOT a shortfall. A page of nothing but replies-to-others means we
    observed this candidate and they said nothing of their own — a fact, not a failure."""
    _candidate(conn, "11", handle="p11")
    out = _run(conn, fake_embedder, monkeypatch,
               {"11": [_tweet("1", reply_to="99"), _tweet("2", reply_to="98")]})
    assert out["by_status"][probe_store.STATUS_OK] == 1
    assert probe_store.count_probe_atoms(conn) == 0


def test_an_unchanged_re_pull_re_embeds_nothing(conn, recording_embedder, monkeypatch):
    def _content_calls() -> int:
        """Embed calls carrying real chunk text. The one-string `probe preflight` probe runs every
        run by design (it is what stops a dead embedder from spending X requests), so counting it
        here would report the hash-skip as broken when it is working."""
        return len([c for c in recording_embedder.calls if c != ["probe preflight"]])

    _candidate(conn, "11", handle="p11")
    tweets = {"11": [_tweet("1", text="an agent framework")]}
    _run(conn, recording_embedder, monkeypatch, tweets)
    after_first = _content_calls()
    assert after_first > 0

    # TTL 0 forces the candidate back into the queue, so the SKIP under test is the hash skip.
    out = _run(conn, recording_embedder, monkeypatch, tweets, ttl_days=0)
    assert _content_calls() == after_first                  # no re-embed
    assert out["by_status"][probe_store.STATUS_OK] == 1     # still a successful observation
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1


# ── the characterization window: why the walk stops on SPAN, not on page count ──
#
# Measured over the 25-account run, ONE page reaches back 1 day for the most prolific account and
# 669 days for the quietest — a 669x spread in the evidence each person is judged on, from an
# identical number of requests. A fixed page count is therefore not a controlled sample, and these
# tests pin the rule that replaced it.

def _dated(tid: str, created: str, text: str = "hello world") -> dict:
    return {"id": tid, "createdAt": created, "text": text, "isRetweet": False, "isReply": False,
            "author": {"id": "9", "userName": "p"}, "url": f"https://x.com/p/status/{tid}"}


def test_span_days_ignores_undated_posts_rather_than_reading_them_as_epoch():
    """A single unparseable timestamp read as 1970 reports a ~20,000-day span and stops the walk on
    its first page — a failure that looks exactly like success. Skipped, not defaulted."""
    from pipeline.kb.candidate_probe import _span_days

    dated = [_dated("1", "Mon Aug 04 12:00:00 +0000 2026"),
             _dated("2", "Mon Aug 11 12:00:00 +0000 2026")]
    assert round(_span_days(dated)) == 7

    with_junk = dated + [_dated("3", ""), _dated("4", "not a date")]
    assert round(_span_days(with_junk)) == 7, "an undated post moved the span"

    assert _span_days([]) == 0.0
    assert _span_days([_dated("1", "Mon Aug 04 12:00:00 +0000 2026")]) == 0.0   # one post = no span


def test_the_walk_stops_once_the_window_is_covered():
    """The quiet account: one page already spans years, so page two is never requested. This is
    where the rule SAVES requests — 14 of 24 measured accounts stop exactly here."""
    from pipeline.kb.candidate_probe import _enough_for_characterization

    stop = _enough_for_characterization(90.0)
    wide = [_dated("1", "Mon Aug 04 12:00:00 +0000 2024"),
            _dated("2", "Mon Aug 11 12:00:00 +0000 2026")]
    assert stop(wide) is True


def test_the_walk_continues_when_the_window_is_too_thin():
    """The prolific account: one page covers a day, which is the case the whole rule exists for."""
    from pipeline.kb.candidate_probe import _enough_for_characterization

    stop = _enough_for_characterization(90.0)
    narrow = [_dated("1", "Mon Aug 10 12:00:00 +0000 2026"),
              _dated("2", "Mon Aug 11 12:00:00 +0000 2026")]
    assert stop(narrow) is False


def test_pagination_paces_itself_and_never_pauses_after_the_last_page():
    """`fetch_user_tweets` deliberately never sleeps, so a MULTI-PAGE caller must pace inside its
    own walk or hand a burst to every other GraphQL consumer sharing the session budget.

    The ordering is the subtle half: the span is checked BEFORE the sleep, so a walk that is
    already finished does not pay for a pause ahead of a request it will never make."""
    import pipeline.kb.candidate_probe as cp

    slept: list[float] = []
    orig = cp.time.sleep
    cp.time.sleep = slept.append
    try:
        thin = [_dated("1", "Mon Aug 10 12:00:00 +0000 2026"),
                _dated("2", "Mon Aug 11 12:00:00 +0000 2026")]
        wide = [_dated("1", "Mon Aug 04 12:00:00 +0000 2024"),
                _dated("2", "Mon Aug 11 12:00:00 +0000 2026")]

        assert cp._enough_for_characterization(90.0, 22.0)(thin) is False
        assert slept == [22.0], "a continuing walk must pace before its next request"

        slept.clear()
        assert cp._enough_for_characterization(90.0, 22.0)(wide) is True
        assert slept == [], "a finished walk paused before a request it never makes"
    finally:
        cp.time.sleep = orig


def test_the_page_cap_still_bounds_a_walk_the_span_never_satisfies():
    """An account posting 50 times a day never reaches 90 days, so the span stop alone would page
    forever. The cap is what makes the rule safe; the span is what makes it useful."""
    from pipeline.ingestion import x_graphql_core as core

    assert core._USERTWEETS_MAX_PAGES >= 1
    from pipeline.kb.candidate_probe import DEFAULT_PAGES, DEFAULT_SPAN_DAYS
    assert DEFAULT_PAGES >= 1 and DEFAULT_SPAN_DAYS > 0
