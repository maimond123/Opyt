"""Oracle screen — ranking and the groups the host prints. Pure over entities +
curation_signals: no model call, no network."""
from __future__ import annotations

import json

import pytest

from pipeline.curation import clock
from pipeline.kb import resolve, schema
from pipeline.oracles import ranking
from pipeline.oracles import screen


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _person(conn, eid, *, name=None, links=None, profile=None):
    schema.upsert_entity(conn, eid, name=name, identity_links=links, profile=profile)


# ── ranking: group-by-canonical + sort ──────────────────────────────────

def test_rank_pools_signals_across_the_canonical_cluster(conn):
    # x:user:1 (X website → carol.substack.com) + substack:carol (home) merge during entity resolution, so a
    # follow on X and a subscribe on Substack must pool into one candidate with distinct=2.
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="Carol Writes", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    schema.ensure_signal(conn, "x:user:1", "follow", "x")
    schema.ensure_signal(conn, "substack:carol", "subscribe", "substack", extra={"is_paid": None})

    cands = ranking.rank_candidates(conn)
    assert len(cands) == 1, [c.canonical_id for c in cands]
    c = cands[0]
    assert c.distinct_signals == 2 and c.has_endorsement
    assert {"x:user:1", "substack:carol"} == set(c.members)
    assert c.name == "Carol"                      # prefers the X row's name


def test_repeated_truncated_following_walks_do_not_retire_candidates(conn):
    """Only an accepted following count may establish the retirement baseline."""
    for i in range(4):
        entity_id = f"x:user:{i}"
        _person(conn, entity_id, name=f"P{i}")
        schema.ensure_signal(conn, entity_id, "follow", "x")

    starts = ["2026-08-12T00:00:00+00:00", "2026-08-12T06:00:00+00:00",
              "2026-08-12T12:00:00+00:00", "2026-08-12T18:00:00+00:00"]
    for started_at, found in zip(starts, (4, 4, 1, 1)):
        clock.record_run(conn, "x_following", status="ok", found=found,
                                  stored_after=4, now=started_at, started_at=started_at)
    conn.execute("UPDATE curation_signals SET last_confirmed_at=? WHERE entity_id='x:user:0'",
                 (starts[-1],))
    conn.execute("UPDATE curation_signals SET last_confirmed_at=? WHERE entity_id!='x:user:0'",
                 (starts[0],))
    conn.commit()

    assert {c.canonical_id for c in ranking.rank_candidates(conn)} == {
        "x:user:0", "x:user:1", "x:user:2", "x:user:3"}


def test_sort_key_is_endorsement_then_distinct_then_count(conn):
    # A: two distinct content signals (bookmark+like), no endorsement, very high count.
    _person(conn, "x:user:A", name="A")
    schema.ensure_signal(conn, "x:user:A", "save", "x", count=12)
    schema.ensure_signal(conn, "x:user:A", "like", "x", count=30)
    # B: one endorsement signal (follow), lowest possible count.
    _person(conn, "x:user:B", name="B")
    schema.ensure_signal(conn, "x:user:B", "follow", "x")
    # C: one content signal only.
    _person(conn, "x:user:C", name="C")
    schema.ensure_signal(conn, "x:user:C", "like", "x", count=3)

    order = [c.canonical_id for c in ranking.rank_candidates(conn)]
    # B wins on endorsement alone, beating A's two content signals and 42 content acts: a follow
    # is a person-level act and no amount of reading outranks it.
    # Then A > C inside the content tier, on distinct signals.
    assert order == ["x:user:B", "x:user:A", "x:user:C"]


def test_variety_beats_volume_inside_the_content_tier(conn):
    _person(conn, "x:user:MIX", name="Mix")
    schema.ensure_signal(conn, "x:user:MIX", "save", "x", count=1)
    schema.ensure_signal(conn, "x:user:MIX", "like", "x", count=1)
    _person(conn, "x:user:VOL", name="Vol")
    schema.ensure_signal(conn, "x:user:VOL", "save", "x", count=5)

    order = [c.canonical_id for c in ranking.rank_candidates(conn)]
    assert order == ["x:user:MIX", "x:user:VOL"]


def test_bookmark_and_like_carry_identical_weight(conn):
    # Neither is an endorsement, so both fall to the same tiebreak and order purely by count.
    _person(conn, "x:user:S3", name="S3")
    schema.ensure_signal(conn, "x:user:S3", "save", "x", count=3)
    _person(conn, "x:user:L2", name="L2")
    schema.ensure_signal(conn, "x:user:L2", "like", "x", count=2)

    order = [c.canonical_id for c in ranking.rank_candidates(conn)]
    assert order == ["x:user:S3", "x:user:L2"]


def test_reflect_degrades_when_paid_unknown(conn):
    _person(conn, "substack:x", name="X")
    schema.ensure_signal(conn, "substack:x", "subscribe", "substack", extra={"is_paid": None})
    c = ranking.rank_candidates(conn)[0]
    assert screen.reflect(c.signals) == "you subscribe"      # not "paid" when unknown
    schema.ensure_signal(conn, "substack:x", "subscribe", "substack", extra={"is_paid": True})
    c2 = ranking.rank_candidates(conn)[0]
    assert "paid subscriber" in screen.reflect(c2.signals)


# ── the groups the host prints ──────────────────────────────

def _cand(cid, *, signals, endorsement=False, distinct=1, count=1):
    return ranking.Candidate(canonical_id=cid, name=cid, signals=signals,
                            has_endorsement=endorsement, distinct_signals=distinct,
                            total_count=count)


def _sig(signal_type, platform, count=1, extra=None):
    return {"signal_type": signal_type, "platform": platform, "count": count, "extra": extra}


_FOLLOW = [_sig("follow", "x")]
_PAPERS = [_sig("save", "openalex", 3)]


def _groups(scr) -> dict:
    return {g["title"]: g for g in scr["groups"]}


def _ids(group) -> list[str]:
    return [c["canonical_id"] for c in group["candidates"]]


@pytest.mark.parametrize("signals, expected", [
    # an endorsement wins categorically, however much content the other source holds
    ([_sig("follow", "substack"), _sig("save", "x", 40), _sig("like", "x", 9)], "substack"),
    # no endorsement anywhere: more distinct signal types wins over volume
    ([_sig("save", "x", 1), _sig("like", "x", 1), _sig("save", "substack", 40)], "x"),
    # same endorsement and variety: the higher total count wins
    ([_sig("save", "x", 2), _sig("save", "substack", 5)], "substack"),
    # identical on both: the first in SOURCE_ORDER, never a coin flip
    ([_sig("subscribe", "substack"), _sig("follow", "x")], "x"),
    # the two paper registries are one source
    ([_sig("save", "openalex", 2), _sig("save", "scholar", 2), _sig("like", "x", 3)], "papers"),
])
def test_stronger_source_table(signals, expected):
    assert ranking.stronger_source(signals) == expected


def test_each_source_is_its_own_group_top_ten_in_rank_order_with_the_rest_counted(
        conn, monkeypatch):
    for i in range(14):
        _person(conn, f"x:user:{i:02d}", name=f"PERSON-X{i}", profile={"bio": "writes"})
        schema.ensure_signal(conn, f"x:user:{i:02d}", "save", "x", count=100 - i)
    for i in range(3):
        _person(conn, f"substack:s{i}", name=f"PERSON-S{i}", profile={"bio": "writes"})
        schema.ensure_signal(conn, f"substack:s{i}", "follow", "substack")

    scr = screen.build_screen(conn)

    assert [g["title"] for g in scr["groups"]] == ["Your top people on X",
                                                    "Your top people on Substack"]
    x, sub = scr["groups"]
    assert _ids(x) == [f"x:user:{i:02d}" for i in range(10)] and x["more"] == 4
    assert _ids(sub) == ["substack:s0", "substack:s1", "substack:s2"] and sub["more"] == 0
    assert (x["source"], sub["source"]) == ("x", "substack")
    assert scr["total_candidates"] == 17


def test_the_more_are_reachable_by_raising_limit_without_moving_a_rank(conn, monkeypatch):
    """The cap bounds what is returned, never what is ranked: a wider page starts with the same
    people in the same order."""
    for i in range(14):
        _person(conn, f"x:user:{i:02d}", name=f"PERSON-{i}", profile={"bio": "writes"})
        schema.ensure_signal(conn, f"x:user:{i:02d}", "save", "x", count=100 - i)

    narrow = screen.build_screen(conn)["groups"][0]
    wide = screen.build_screen(conn, limit=20)["groups"][0]

    assert _ids(wide)[:10] == _ids(narrow) and len(_ids(wide)) == 14 and wide["more"] == 0


def test_a_person_on_two_sources_is_listed_once_under_the_stronger_with_also_on(
        conn, monkeypatch):
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="Carol", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    schema.ensure_signal(conn, "x:user:1", "save", "x", count=35)
    schema.ensure_signal(conn, "x:user:1", "like", "x", count=3)
    schema.ensure_signal(conn, "substack:carol", "save", "substack", count=2)

    scr = screen.build_screen(conn)
    cards = [c for g in scr["groups"] for c in g["candidates"]]

    assert len(cards) == 1 and [g["title"] for g in scr["groups"]] == ["Your top people on X"]
    assert cards[0]["reason"] == "bookmarked 35, liked 3, also on Substack"


def test_organisations_stay_in_their_source_group_and_the_host_is_told_to_move_them(
        conn, monkeypatch):
    """No model call decides what an account is. An organisation is
    listed with the people, in rank order, and the note tells the host to move plain ones into one
    last group."""
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "call", lambda *a, **k: pytest.fail("the screen called a model"))
    _person(conn, "x:user:p", name="PERSON-1", profile={"bio": "builder"})
    schema.ensure_signal(conn, "x:user:p", "follow", "x")
    _person(conn, "x:user:o", name="ORG-1", profile={"bio": "the company"})
    schema.ensure_signal(conn, "x:user:o", "follow", "x")
    schema.ensure_signal(conn, "x:user:o", "list", "x", count=6, extra={"list_names": ["Research"]})

    scr = screen.build_screen(conn)

    assert [g["title"] for g in scr["groups"]] == ["Your top people on X"]
    assert _ids(scr["groups"][0]) == ["x:user:o", "x:user:p"]
    assert "Organisations, media and projects" in scr["note"]
    assert "when unsure, leave it" in scr["note"]
    assert "classify" not in scr


def test_a_paper_only_author_has_their_own_group_and_nobody_is_selected(conn, monkeypatch):
    """A paper registry has no follow primitive, so under one flat order a researcher sat below
    every follow. Their own group fixes that without weakening the endorsement key."""
    schema.upsert_entity(conn, "x:user:1", name="PERSON-1")
    schema.ensure_signal(conn, "x:user:1", "follow", "x")
    schema.upsert_entity(conn, "openalex:A1", name="F. Arnold")
    schema.set_signal(conn, "openalex:A1", "save", "openalex", count=9)
    schema.set_signal(conn, "openalex:A1", "save", "scholar", count=4)

    groups = _groups(screen.build_screen(conn))

    papers = groups["Your top people from papers"]
    assert _ids(papers) == ["openalex:A1"] and papers["source"] == "papers"
    assert set(papers["candidates"][0]) == {"canonical_id", "name", "handle", "reason"}


def test_a_person_the_user_also_follows_is_not_a_paper_only_author():
    """All signals scholar, not any. Someone the user follows on X who also wrote a paper they
    saved has a real endorsement, and it should earn them their rank under X."""
    both = _cand("x:9", signals=_FOLLOW + _PAPERS, endorsement=True, distinct=2)
    assert both.is_scholar is False
    assert _cand("openalex:A1", signals=_PAPERS).is_scholar is True
    assert ranking.stronger_source(both.signals) == "x"


def test_the_authors_of_saved_papers_are_reflected_as_papers_not_posts():
    """A non-X save is not assumed to be a Substack post: an author of three saved papers is
    reflected back as "saved 3 papers", not "saved 3 posts"."""
    assert screen.reflect(_PAPERS) == "saved 3 papers"
    assert screen.reflect([_sig("save", "x", 2)]) == "bookmarked 2"
    assert screen.reflect([_sig("save", "substack", 1)]) == "saved 1 post"


def test_the_reason_puts_endorsements_first_whatever_order_the_rows_come_in():
    assert screen.reflect([_sig("like", "x", 3), _sig("save", "x", 35), _sig("follow", "x")]) \
        == "you follow, bookmarked 35, liked 3"


def test_the_note_tells_the_host_to_print_and_ask_and_says_nothing_of_a_pre_tick(
        conn, monkeypatch):
    _person(conn, "x:user:1", name="PERSON-1")
    schema.ensure_signal(conn, "x:user:1", "follow", "x")

    scr = screen.build_screen(conn)

    for said in (scr["note"], scr["ask"]):
        for word in ("pre-tick", "pre_tick", "corroborat", "recommended", "shown by default"):
            assert word not in said.lower()
    assert scr["ask"].startswith("Pick the people you want as Oracles")
    assert "about 5" in scr["ask"] and "add more any time" in scr["ask"]
    assert not {"recommended_count", "shown_by_default_count", "floor", "candidates"} & set(scr)


# ── the payload a host can actually read ──────────────────────────────────────

def test_a_real_sized_store_fits_in_the_reader(conn, monkeypatch):
    """The measurement, not the mechanism. Payloads of 68KB and 52KB were too large for a host
    to read. Any change to `_card` that pushes a default screen over ~25KB reopens that."""
    for i in range(400):
        eid = f"x:user:{i}" if i % 2 else f"substack:s{i}"
        _person(conn, eid, name=f'{("PERSON", "ORG")[i % 4 == 1]}-{i}',
                profile={"bio": "a writer of things"})
        schema.ensure_signal(conn, eid, "follow", "x" if i % 2 else "substack")
        schema.ensure_signal(conn, eid, "like", "x" if i % 2 else "substack", count=400 - i)

    size = len(json.dumps(screen.build_screen(conn)))

    assert size < 25_000, f"a default screen is {size:,} chars — the host could not read 52,000"


def test_a_card_carries_the_sentence_and_not_the_rows_behind_it(conn, monkeypatch):
    """`signals`, `identity_links` and `members` are the raw form of something the card already
    states, so the card carries only the sentence. Each extra field is something the host would
    feel bound to explain."""
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    schema.ensure_signal(conn, "x:user:1", "follow", "x")
    schema.ensure_signal(conn, "x:user:1", "save", "x", count=12)

    card = screen.build_screen(conn)["groups"][0]["candidates"][0]

    assert card == {"canonical_id": "x:user:1", "name": "Carol", "handle": None,
                    "reason": "you follow, bookmarked 12"}


# ── source= : the bounded form of "show me more of my Substack people" ──────────
#
# Without it, a host asked for someone's Substack writers calls `screen(limit=1100)`, which
# returned 290,754 characters, over the token limit. The cap was not the problem; the missing
# question was.
def _followed(conn, eid, *, name, count=1):
    _person(conn, eid, name=name, profile={"bio": "writes"})
    schema.ensure_signal(conn, eid, "save", "x", count=count)


def _both_platforms(conn, eid, *, name):
    """Someone the user follows on X and subscribes to on Substack."""
    _person(conn, eid, name=name, profile={"bio": "writes"})
    schema.ensure_signal(conn, eid, "follow", "x")
    schema.ensure_signal(conn, eid, "subscribe", "substack")


def test_source_narrows_who_is_listed(conn, monkeypatch):
    _followed(conn, "x:user:1", name="X-ONLY")
    _person(conn, "substack:sub", name="SUBSTACK-ONLY", profile={"bio": "writes"})
    schema.ensure_signal(conn, "substack:sub", "subscribe", "substack")

    out = screen.build_screen(conn, source="substack")

    assert [c["name"] for g in out["groups"] for c in g["candidates"]] == ["SUBSTACK-ONLY"]
    assert out["total_candidates"] == 1 and out["source"] == "substack"


def test_source_is_membership_not_exclusivity(conn, monkeypatch):
    """Asked for Substack, someone on X and Substack is listed under Substack — the group without
    `source=` lists them under X, the stronger, and that must not hide them here."""
    _both_platforms(conn, "x:user:1", name="BOTH")

    sub = screen.build_screen(conn, source="substack")["groups"]

    assert [(g["title"], [c["name"] for c in g["candidates"]]) for g in sub] == [
        ("Your top people on Substack", ["BOTH"])]
    assert sub[0]["candidates"][0]["reason"] == "you subscribe, also on X"


def test_source_takes_the_papers_group_by_its_own_name(conn, monkeypatch):
    schema.upsert_entity(conn, "openalex:A1", name="F. Arnold")
    schema.set_signal(conn, "openalex:A1", "save", "openalex", count=3)
    _followed(conn, "x:user:1", name="PERSON-1")

    out = screen.build_screen(conn, source="papers")

    assert [g["title"] for g in out["groups"]] == ["Your top people from papers"]


def test_source_narrows_the_list_without_moving_a_rank(conn, monkeypatch):
    """`source=` is a filter, never a re-score. Filtering after ranking is what guarantees it."""
    for i in range(6):
        _person(conn, f"substack:s{i}", name=f"S-{i}", profile={"bio": "writes"})
        schema.ensure_signal(conn, f"substack:s{i}", "subscribe", "substack", count=50 - i)
    _both_platforms(conn, "x:user:b", name="BOTH")

    subs = _ids(screen.build_screen(conn, source="substack")["groups"][0])
    ranked = [c.canonical_id for c in ranking.rank_candidates(conn)]

    assert subs == [cid for cid in ranked if cid in set(subs)], "filtering reordered the list"


def test_an_unknown_source_says_so_instead_of_returning_nobody(conn, monkeypatch):
    """An empty list is a guess here: it reads exactly like a connected platform nobody arrives
    through. Naming what the store holds is the repair."""
    _followed(conn, "x:user:1", name="PERSON-1")

    out = screen.build_screen(conn, source="twitter")

    assert "twitter" in out["error"] and "x" in out["known_platforms"]
    assert "groups" not in out


def test_an_oversized_limit_is_clamped_to_the_total_and_reported_not_honored(conn, monkeypatch):
    """A documented "do not" the tool cheerfully honors is not a guard — the host raised `limit`
    to 1100 precisely because nothing stopped it. The bound is a TOTAL across the groups."""
    for i in range(120):
        _followed(conn, f"x:user:{i:03d}", name=f"PERSON-X{i}", count=200 - i)
        _person(conn, f"substack:s{i:03d}", name=f"PERSON-S{i}", profile={"bio": "writes"})
        schema.ensure_signal(conn, f"substack:s{i:03d}", "follow", "substack")

    out = screen.build_screen(conn, limit=1100)

    half = screen.SCREEN_LIMIT_MAX // 2
    assert [len(g["candidates"]) for g in out["groups"]] == [half, half]
    assert [g["more"] for g in out["groups"]] == [120 - half, 120 - half]
    assert out["limit_clamped"] == {"asked": 1100, "applied": half,
                                    "note": out["limit_clamped"]["note"]}
    assert "source=" in out["limit_clamped"]["note"], "say how to ask the bounded question"


def test_an_ordinary_limit_reports_no_clamp(conn, monkeypatch):
    _followed(conn, "x:user:1", name="PERSON-1")
    assert "limit_clamped" not in screen.build_screen(conn)


def test_a_confirmed_oracle_is_not_waiting_to_be_screened(conn, monkeypatch):
    """A confirmed Oracle is not ranked, counted as "waiting to be screened" or shown on the
    screen. One rule in `ranking`: a waiting candidate is off the roster, under any anchor of
    its cluster."""
    from pipeline.oracles import roster

    for uid in ("1", "2"):
        _person(conn, f"x:user:{uid}", name=f"P{uid}")
        schema.ensure_signal(conn, f"x:user:{uid}", "follow", "x")
    roster.upsert_oracle(conn, "x:user:1", name="P1")
    _person(conn, "x:user:3", name="P1")
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:3", "x:user:3": "x:user:3"})

    assert [c.canonical_id for c in ranking.rank_candidates(conn)] == ["x:user:2"]
    assert ranking.waiting_count(conn) == 1
    shown = [c["canonical_id"] for g in screen.build_screen(conn, limit=10)["groups"]
             for c in g["candidates"]]
    assert "x:user:3" not in shown and "x:user:1" not in shown


# ── "Named by your Oracles" ─────────────────────────────────

def test_a_named_candidate_stays_in_their_group_and_the_rest_keep_their_rank():
    ranked = [ranking.Candidate(canonical_id="p:1", members=["x:user:1", "substack:one"])]
    named = [{"who_id": "x:user:9", "oracles": ["a"]},
             {"who_id": "x:user:1", "oracles": ["a", "b"]},
             {"who_id": "x:user:8", "oracles": ["b"]}]

    times, newcomers = screen.split_named(named, ranked)

    assert times == {"p:1": 2}
    assert [r["who_id"] for r in newcomers] == ["x:user:9", "x:user:8"]


def _engage(kind, handle, uid=None, words="w"):
    return {"kind": kind, "who": f"x:user:{uid}" if uid else None, "handle": handle,
            "words": words, "url": f"https://x.com/o/status/{handle}-{kind}"}


def _oracle(conn, uid, handle):
    from pipeline.oracles import roster
    schema.upsert_entity(conn, f"x:user:{uid}", name=handle.title(), profile={"handle": handle})
    conn.execute("UPDATE entities SET canonical_id = entity_id WHERE entity_id = ?",
                 (f"x:user:{uid}",))
    roster.upsert_oracle(conn, f"x:user:{uid}", handle.title())


def _posts(conn, posts):
    """`posts` = [(author uid, text, engages)] → Oracle footprint atoms through the one writer."""
    from pipeline.kb.ingest_common import store_atom
    for i, (uid, text, engages) in enumerate(posts):
        store_atom(conn, {"atom_id": f"xprofile:{i}", "source_type": "x",
                          "who_id": f"x:user:{uid}", "entry_mode": "oracle-footprint",
                          "description": "d", "raw_hash": f"h{i}",
                          "when_ts": f"2026-10-0{i % 9 + 1}", "payload": {"engages": engages}},
                   text)


def test_people_the_oracles_name_are_a_last_group_once_oracles_exist(conn):
    """Read live from the Oracles' posts, never written as curation signals: a signal row would
    queue every named person for the probe scan."""

    _person(conn, "x:user:1", name="PERSON-1")
    schema.ensure_signal(conn, "x:user:1", "follow", "x")
    assert [g["title"] for g in screen.build_screen(conn)["groups"]] == ["Your top people on X"]

    _oracle(conn, 100, "soren")
    _oracle(conn, 200, "imas")
    _posts(conn, [
        (100, "one", [_engage("mention", "person1", "1"), _engage("quote", "bob", "7")]),
        (200, "two", [_engage("mention", "person1", "1")]),
    ])

    scr = screen.build_screen(conn)
    x, named = scr["groups"]
    assert x["candidates"][0]["reason"] == "you follow, named by 2 of your Oracles"
    assert (named["title"], named["source"]) == ("Named by your Oracles", "named")
    assert named["candidates"] == [{"canonical_id": None, "name": "bob", "handle": "bob",
                                    "reason": "named by one of your Oracles"}]
    assert scr["total_candidates"] == 2
    assert not conn.execute("SELECT 1 FROM curation_signals WHERE entity_id = 'x:user:7'").fetchone()
    assert [g["source"] for g in screen.build_screen(conn, source="named")["groups"]] == ["named"]
    assert [g["source"] for g in screen.build_screen(conn, source="x")["groups"]] == ["x"]
