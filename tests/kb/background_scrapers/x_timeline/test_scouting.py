"""Scouting (`background_scrapers.x_timeline.scouting`): one page per unit, three pages or 90
days per candidate, sampled once, and failures charged like everything else."""
from __future__ import annotations

from datetime import timedelta

from pipeline.oracles import probe_store
from pipeline.oracles.sources.timeline import samples, sources
from pipeline.oracles.sources.timeline.scouting import Scouting
from pipeline.platforms.x.originals import XUserUnavailable
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop.retries import Wait

from .._support import NOW, days, pair, sample, stored


def test_scouting_waits_for_an_oracle(conn, scout):
    assert Scouting().next_unit(conn, NOW) is None
    pair(conn)
    assert Scouting().next_unit(conn, NOW).key == "x:user:5"
    assert Scouting.floor == 50


def test_a_sample_is_three_pages_each_its_own_unit_and_never_taken_again(conn, loop, ctx, scout):
    served, calls = scout
    pair(conn)
    served += [sample("90", days(10)), sample("80", days(20)), sample("70", days(30))]
    scouting = Scouting()
    results = [loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result
               for _ in range(3)]
    assert results == ["sampling", "sampling", "ok"]
    assert calls == [None, "90", "80"]
    row = probe_store.pull_states(conn)["x:user:5"]
    assert (row["pages_done"], row["atoms"]) == (3, 6)
    assert scouting.next_unit(conn, NOW + timedelta(days=400)) is None   # no re-sample, ever


def test_a_sample_stops_at_ninety_days(conn, loop, ctx, scout):
    served, _calls = scout
    pair(conn)
    served.append(sample("90", days(120)))
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "ok"


def test_a_failed_sample_is_retried_later_from_where_it_stopped(conn, loop, ctx, scout):
    served, calls = scout
    pair(conn)
    served += [sample("90", days(10)), RuntimeError("timeout"), sample("80", days(20))]
    scouting = Scouting()
    loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW))
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "blip"
    row = probe_store.pull_states(conn)["x:user:5"]
    retry = parse_ts(row["retry_at"])
    assert retry > NOW and scouting.next_unit(conn, NOW) is None
    assert scouting.next_due(conn, NOW) == retry
    loop.run_unit(conn, scouting, scouting.next_unit(conn, retry))
    assert calls == [None, "90", "90"]


def test_scouting_failures_count_toward_x_failing_and_are_given_back(conn, loop, ctx, scout, walk,
                                                                     no_jitter):
    served, _calls = walk
    sampled, _scalls = scout
    pair(conn, "alice")
    pair(conn, "bob")
    served += [RuntimeError("timeout"), RuntimeError("timeout")]
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    sampled.append(RuntimeError("timeout"))
    scouting = Scouting()
    paused = loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW))
    assert isinstance(paused, Wait) and (paused.state, paused.cause) == ("paused", "x_outage")
    assert scouting.next_unit(conn, NOW) is not None        # still owed, and due now
    assert stored(conn, "alice").failures is None


def test_an_unavailable_account_is_final(conn, loop, ctx, scout):
    served, _calls = scout
    pair(conn)
    served.append(XUserUnavailable("suspended"))
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "unavailable"
    assert scouting.next_unit(conn, NOW) is None




def test_a_candidate_that_failed_ten_times_is_never_sampled_again(conn, loop, ctx, scout, no_jitter):
    served, _calls = scout
    pair(conn)
    probe_store.init_probe_schema(conn)
    probe_store.record_sample(conn, "x:user:5", probe_store.STATUS_FAILED, atoms=0, pages_done=0,
                              tail_id=None, retry_at=NOW.isoformat(), failures=9)
    served.append(RuntimeError("timeout"))
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "blip"
    row = probe_store.pull_states(conn)["x:user:5"]
    assert row["failures"] == 10 and row["retry_at"] is None
    assert scouting.next_unit(conn, NOW + timedelta(days=400)) is None
    assert scouting.next_due(conn, NOW) is None


def test_an_account_with_nothing_posted_is_empty_not_failed(conn, loop, ctx, scout):
    """A durable fact, never re-fetched or read as "no field"."""
    served, _calls = scout
    pair(conn)
    served.append(sample(None, None, ended=True, fetched=0, written=0))
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == probe_store.STATUS_EMPTY


def test_a_page_with_few_posts_of_their_own_is_still_a_page_read(conn, loop, ctx, scout):
    """A page whose posts were mostly replies to others stores few atoms, and that is an
    observation, not a failure: the sample moves on with what the page held."""
    served, calls = scout
    pair(conn)
    served += [sample("90", days(10), written=1), sample("80", days(20), written=0)]
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "sampling"
    row = probe_store.pull_states(conn)["x:user:5"]
    assert (row["status"], row["atoms"], row["pages_done"]) == (
        probe_store.STATUS_SAMPLING, 1, 1)
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "sampling"
    assert calls == [None, "90"]


def test_a_page_and_the_candidates_standing_commit_together(conn, loop, ctx, scout, monkeypatch):
    """The page's atoms are written in the same transaction as `probe_pulls`: when the lease goes
    while the page was read, neither moves, and the page is read again later."""
    served, _calls = scout
    pair(conn)
    wrote = []
    monkeypatch.setattr(samples, "write_sample", lambda conn, page: wrote.append(page))
    first = sample("90", days(10))
    served.append(first)

    loop.ctx.lease_lost = lambda: not served              # lost once the page has been served
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "lease_lost"
    assert wrote == []
    assert probe_store.pull_states(conn).get("x:user:5", {}).get("pages_done") in (None, 0)

    loop.ctx.lease_lost = lambda: False
    served.append(first)
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "sampling"
    assert wrote == [first]
    assert probe_store.pull_states(conn)["x:user:5"]["pages_done"] == 1


def test_a_handle_declared_on_substack_is_resolved_and_merged_once(conn, loop, ctx, monkeypatch):
    """A Substack-only candidate whose byline names an X handle: scouting looks it up once, mints
    the X account attesting the publication, and the next queue has them by id."""
    from pipeline.kb import schema
    from pipeline.oracles import roster
    from pipeline.platforms.x import profile as x_profile
    roster.upsert_oracle(conn, "x:user:1", name="One")
    schema.upsert_entity(conn, "substack:nine", name="Nine",
                         identity_links=["https://nine.substack.com"],
                         profile={"x_handle": "NineWrites"})
    schema.ensure_signal(conn, "substack:nine", "subscribe", "substack")
    pair(conn)
    asked = []
    monkeypatch.setattr(x_profile, "fetch_user_profile", lambda s, h: asked.append(h) or {
        "user_id": "77", "handle": "NineWrites", "display_name": "Nine", "bio": "",
        "verified": False, "followers": 10})

    assert [c["who_id"] for c in samples.candidate_queue(conn)] == ["x:@ninewrites"]
    scouting = Scouting()
    assert loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW)).result == "resolved"
    assert asked == ["NineWrites"]
    assert schema.current_canonical(conn, "x:user:77") == schema.current_canonical(
        conn, "substack:nine")
    queued = samples.candidate_queue(conn)
    assert [c["who_id"] for c in queued] == ["x:user:77"]
    assert scouting.next_unit(conn, NOW).key == "x:user:77"


def test_a_declared_handle_x_cannot_read_is_final(conn, loop, ctx, monkeypatch):
    from pipeline.kb import schema
    from pipeline.oracles import roster
    from pipeline.platforms.x import profile as x_profile
    roster.upsert_oracle(conn, "x:user:1", name="One")
    schema.upsert_entity(conn, "substack:nine", name="Nine",
                         identity_links=["https://nine.substack.com"],
                         profile={"x_handle": "gone"})
    schema.ensure_signal(conn, "substack:nine", "subscribe", "substack")
    pair(conn)
    monkeypatch.setattr(x_profile, "fetch_user_profile", lambda s, h: None)
    scouting = Scouting()
    loop.run_unit(conn, scouting, scouting.next_unit(conn, NOW))
    assert probe_store.pull_states(conn)["x:@gone"]["status"] == "unavailable"
    assert scouting.next_unit(conn, NOW) is None
