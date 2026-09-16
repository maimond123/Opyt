"""The topic filter on a scholar Oracle, and the venue root that rides on the same code.

Two things are being pinned here, and only one of them is the filter itself.

The FIRST is that the selection lives on the `oracle_sources` row rather than on the ingest call.
`oracle_refresh` re-pulls a scholar pair forever from `source_key`, so a filter applied only at
ingest time widens back to the whole corpus within one TTL — silently, because nothing reports a
pull that got MORE than it was asked for. `test_seeding_after_an_ingest_does_not_erase_the_choice`
and `test_the_refresh_rail_pulls_through_the_stored_filter` are that regression.

The SECOND is that an author feed and a venue feed are ONE code path. `works_filter` reads the
letter OpenAlex prefixes every id with and picks the `/works` field; nothing else branches on it.

Offline throughout: every adapter method that would leave the process is monkeypatched.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from pipeline.kb import frontier_sources as fs
from pipeline.kb import oracle_refresh_state as st
from pipeline.kb import oracles, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = st.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _scholar_oracle(conn, eid="openalex:A5043841592", name="F. Arnold"):
    schema.upsert_entity(conn, eid, name=name)
    schema.upsert_oracle(conn, eid, name=name)
    conn.commit()
    return eid


# ── one filter builder, two base clauses ────────────────────────────────────────

def test_the_id_prefix_picks_the_base_clause():
    """The whole generalization. An author and a venue differ by one letter, and that letter is
    what makes them the same pull rather than two adapters."""
    assert fs.works_filter("A5043841592") == "author.id:A5043841592"
    assert fs.works_filter("S4393918830") == "primary_location.source.id:S4393918830"


def test_topics_and_dates_are_identical_on_both_bases():
    """Neither extra clause knows which base it is attached to — which is why the venue branch
    needed no filter code of its own."""
    for oid in ("A5043841592", "S4393918830"):
        filt = fs.works_filter(oid, topics="T10404|T11611",
                               since=datetime(2024, 3, 1, tzinfo=timezone.utc))
        assert ",primary_topic.id:T10404|T11611" in filt
        assert ",from_publication_date:2024-03-01" in filt


def test_an_id_with_no_works_field_refuses_rather_than_returning_nothing():
    """`pair_from_member` splits an `openalex:{tail}` entity id and returns the tail unchecked, so
    a malformed entity id reaches here. `author.id:W123` is a well-formed request that returns
    zero works, which reads as "this person published nothing" — the worst lie available."""
    with pytest.raises(ValueError):
        fs.works_filter("W2741809807")


# ── the stored decision ─────────────────────────────────────────────────────────

def test_a_bad_topic_id_is_dropped_and_the_rest_still_narrow(conn):
    """The list is a host echoing ids back at us. One garbled entry should narrow to the rest
    rather than fail the ingest, and nothing but `T\\d+` reaches a URL."""
    cid = _scholar_oracle(conn)
    st.upsert_source(conn, st.SourceRow(cid, "openalex", "A5043841592"))

    stored = st.set_topic_filter(conn, cid, "A5043841592",
                                 ["T10404", "not-a-topic", "T11611", ""])

    assert stored == "T10404|T11611"


def test_an_empty_list_clears_the_filter_and_omitting_it_leaves_it_alone(conn):
    """Two distinct states with a caller each. `[]` is the only way back to the whole corpus;
    NOT calling this is what an ordinary top-up does, and a top-up must not unfilter someone."""
    cid = _scholar_oracle(conn)
    st.upsert_source(conn, st.SourceRow(cid, "openalex", "A5043841592"))
    st.set_topic_filter(conn, cid, "A5043841592", ["T10404"])

    assert st.topic_filter_for(conn, cid, "A5043841592") == "T10404"
    assert st.set_topic_filter(conn, cid, "A5043841592", []) is None
    assert st.topic_filter_for(conn, cid, "A5043841592") is None


def test_seeding_after_an_ingest_does_not_erase_the_choice(conn):
    """THE regression this feature is most likely to reintroduce. `seed_from_entities` rebuilds
    every pair from `entities`, which carry no topic selection, and it runs after EVERY ingest —
    so an overwriting upsert would erase the user's subjects on their next top-up."""
    cid = _scholar_oracle(conn)
    st.seed_from_entities(conn, [cid])
    st.set_topic_filter(conn, cid, "A5043841592", ["T10404", "T11611"])

    st.seed_from_entities(conn, [cid])          # what every later ingest re-runs

    assert st.topic_filter_for(conn, cid, "A5043841592") == "T10404|T11611"


def test_the_filter_survives_a_recorded_pull(conn):
    """`record_pull` writes the freshness columns on the same row. It must not touch this one."""
    cid = _scholar_oracle(conn)
    st.seed_from_entities(conn, [cid])
    st.set_topic_filter(conn, cid, "A5043841592", ["T10404"])

    st.record_pull(conn, st.SourceRow(cid, "openalex", "A5043841592"),
                   last_status="ingested", cursor_ts="2026-09-08T00:00:00+00:00")

    assert st.topic_filter_for(conn, cid, "A5043841592") == "T10404"


def test_a_store_written_before_the_column_existed_migrates_in_place(conn, tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does not add a column, so the ALTER is what carries a live
    store forward — the same shape `covered_from` uses."""
    conn.execute("DROP TABLE oracle_sources")
    conn.execute("CREATE TABLE oracle_sources (canonical_id TEXT NOT NULL, "
                 "source_type TEXT NOT NULL, source_key TEXT NOT NULL, "
                 "status TEXT NOT NULL DEFAULT 'trusted', added_at TEXT, last_pulled_at TEXT, "
                 "cursor_ts TEXT, last_status TEXT, "
                 "PRIMARY KEY (canonical_id, source_type, source_key))")
    conn.commit()

    st.init_state_schema(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(oracle_sources)")}
    assert {"covered_from", "topic_filter"} <= cols


# ── both pulls read the one home ────────────────────────────────────────────────

def test_the_refresh_rail_pulls_through_the_stored_filter(conn, monkeypatch):
    """The non-negotiable half. `oracle_refresh` is the loop that runs forever with nobody
    watching, so it is the one that must not widen."""
    from pipeline.kb import oracle_refresh as orf

    seen = {}

    def _fake(conn_, embedder, **kw):
        seen.update(kw)
        return {"source": "openalex", "atoms": 0, "papers": 0}

    monkeypatch.setattr(orf, "ingest_scholar_footprint_sync", _fake)
    row = st.SourceRow("openalex:A5043841592", "openalex", "A5043841592",
                       topic_filter="T10404|T11611")

    orf._dispatch(conn, object(), row, None, "F. Arnold")

    assert seen["topics"] == "T10404|T11611"
    assert seen["openalex_id"] == "A5043841592"


def test_the_first_backlog_reads_the_same_row_the_rail_does(conn, monkeypatch):
    """One home, two readers. `_ingest_oracle` stores the choice BEFORE pulling, and the pull
    looks it up rather than being handed it — so a backlog and a refresh six months later cannot
    disagree about what the user picked."""
    from pipeline.kb import ingest_scholar_footprint as isf

    cid = _scholar_oracle(conn)
    seen = {}

    def _fake(conn_, embedder, **kw):
        seen.update(kw)
        return {"source": "openalex", "atoms": 0, "papers": 0}

    monkeypatch.setattr(isf, "sync_scholar_footprint", _fake)
    oracles._store_scholar_topics(conn, cid, ["T10404"])

    oracles._pull_scholar_papers(conn, object(), {"name": "F. Arnold"}, cid,
                                 timer=_Timer(), since=None)

    assert seen["topics"] == "T10404"
    assert st.topic_filter_for(conn, cid, "A5043841592") == "T10404"


class _Timer:
    """The two lines of `StageTimer` this path touches."""
    def stage(self, _name):
        import contextlib
        return contextlib.nullcontext()


# ── the count-first ask ─────────────────────────────────────────────────────────

def test_the_year_counts_are_reported_through_the_stored_filter(conn, monkeypatch):
    """Once a user has narrowed, every number read back to them has to describe the pull that
    will run. Reporting 928 against a filter that takes 128 breaks the one thing counts are for."""
    cid = _scholar_oracle(conn)
    st.seed_from_entities(conn, [cid])
    st.set_topic_filter(conn, cid, "A5043841592", ["T10404"])
    y = date.today().year
    passed = {}

    def _counts(self, oid, *, topics=None):
        passed["topics"] = topics
        return {y: 4, y - 1: 6}

    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts", _counts)

    out = oracles.scholar_year_counts(conn, cid)

    assert passed["topics"] == "T10404"
    assert out["total"] == 10 and out["topics"] == "T10404"


def test_the_topic_ask_shows_a_bounded_list_and_states_the_rest(conn, monkeypatch):
    """174 subjects is not a list a host can read aloud. The bound is on DISPLAY only — nothing
    is dropped from the pull by not being named, because everything stays selected."""
    cid = _scholar_oracle(conn)
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "topic_counts",
                        lambda self, oid: [{"id": f"T{i}", "name": f"topic {i}", "count": 200 - i}
                                           for i in range(40)])

    out = oracles.scholar_topic_counts(conn, cid)

    assert len(out["topics"]) == oracles.TOPIC_SHOW
    assert out["more"] == 40 - oracles.TOPIC_SHOW
    assert out["n_topics"] == 40
    assert out["selected"] is None


def test_a_failed_count_leaves_the_ingest_unfiltered_rather_than_blocked(conn, monkeypatch):
    """Fail-safe in the WIDE direction. A count is what makes the question better; it must never
    be what makes it impossible, and it must never silently narrow a pull."""
    cid = _scholar_oracle(conn)

    def _boom(self, oid):
        raise RuntimeError("openalex down")

    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "topic_counts", _boom)

    assert oracles.scholar_topic_counts(conn, cid) is None


def test_the_topic_list_is_not_narrowed_by_what_is_already_selected(conn, monkeypatch):
    """This is the list the user picks FROM. Filtering it to their current choice would hide
    every subject they might add back."""
    cid = _scholar_oracle(conn)
    st.seed_from_entities(conn, [cid])
    st.set_topic_filter(conn, cid, "A5043841592", ["T1"])
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "topic_counts",
                        lambda self, oid: [{"id": "T1", "name": "a", "count": 9},
                                           {"id": "T2", "name": "b", "count": 4}])

    out = oracles.scholar_topic_counts(conn, cid)

    assert [t["id"] for t in out["topics"]] == ["T1", "T2"]
    assert out["selected"] == ["T1"]


# ── the venue root ──────────────────────────────────────────────────────────────

def test_a_venue_resolves_when_exactly_one_source_matches(monkeypatch):
    """Measured 2026-09-08: `chemrxiv` and `biorxiv` each match one OpenAlex source."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name",
                        lambda self, n: [{"id": "https://openalex.org/S4393918830",
                                          "display_name": "ChemRxiv", "works_count": 63600}])

    out = oracles._openalex_root("https://chemrxiv.org")

    assert out["openalex_id"] == "S4393918830"
    assert out["name"] == "ChemRxiv"


@pytest.mark.parametrize("hits", [[], [{"id": "https://openalex.org/S1", "display_name": "a"},
                                       {"id": "https://openalex.org/S2", "display_name": "b"}]])
def test_zero_or_several_matches_stay_a_blog(monkeypatch, hits):
    """The uniqueness rule, and the reason it is "exactly one" rather than "take the top hit".
    Measured 2026-09-08: fourteen of fifteen real blog hosts match ZERO sources, while `medium`
    matches 24 (top hit: a medieval studies journal) and `nature` matches 222."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name", lambda self, n: hits)

    assert oracles._openalex_root("https://simonwillison.net") is None


def test_a_failed_lookup_leaves_the_url_a_blog(monkeypatch):
    """Fail-safe to the behaviour that existed before this function. A venue OPYT cannot resolve
    is still a readable site; a blog wrongly taken for a venue would pull a stranger's corpus."""
    def _boom(self, n):
        raise RuntimeError("openalex down")

    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name", _boom)

    assert oracles._openalex_root("https://chemrxiv.org") is None


def test_a_pasted_openalex_source_url_needs_no_search(monkeypatch):
    """The exact form the ambiguous-venue refusal points at, so a user told to paste one has
    somewhere to paste it."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "source",
                        lambda self, sid: {"id": f"https://openalex.org/{sid}",
                                           "display_name": "ChemRxiv", "works_count": 63600})
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name",
                        lambda self, n: pytest.fail("a pasted id must not be name-searched"))

    assert oracles._openalex_root("https://openalex.org/S4393918830")["openalex_id"] == "S4393918830"


def test_a_venue_mints_the_same_entity_shape_a_researcher_does(conn, monkeypatch):
    """A venue is a filter on OpenAlex, never a new ingester — so it inherits the scholar pull,
    DOI dedup and the topic filter with no adapter of its own. Before this it became
    `blog:chemrxiv.org` and a preprint repository was scraped as generic articles."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name",
                        lambda self, n: [{"id": "https://openalex.org/S4393918830",
                                          "display_name": "ChemRxiv", "works_count": 63600}])

    cid = oracles._resolve_handle(conn, "https://chemrxiv.org")

    assert cid == "openalex:S4393918830"
    assert st.pair_from_member(schema.get_entity(conn, cid)) == ("openalex", "S4393918830")


def test_the_publisher_label_is_searched_not_the_subdomain(monkeypatch):
    """`carol.substack.com` must search "substack", never "carol". Searching the first label puts
    a PERSON'S NAME against a journal index — the wrong question of the wrong corpus, and the one
    way a personal newsletter could resolve to an obscure journal. Measured 2026-09-08:
    `substack`, `github` and `wordpress` all match zero OpenAlex sources, so every blog platform
    stays a blog under this rule without anyone maintaining a host list."""
    asked = []
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name",
                        lambda self, n: asked.append(n) or [])

    for url in ("https://chemrxiv.org", "https://carol.substack.com",
                "https://journals.sagepub.com", "https://www.biorxiv.org"):
        oracles._openalex_root(url)

    assert asked == ["chemrxiv", "substack", "sagepub", "biorxiv"]


def test_an_open_breaker_skips_the_lookup_entirely(monkeypatch):
    """`available()` before the call, the way `frontier_execute` does it — a host already known to
    be down must not cost a round trip per pasted URL to rediscover that."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "available", lambda self: False)
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name",
                        lambda self, n: pytest.fail("the breaker is open; nothing should be sent"))

    assert oracles._openalex_root("https://chemrxiv.org") is None


def test_a_personal_site_is_still_a_blog(conn, monkeypatch):
    """The venue check sits in front of the blog fallthrough, so this is the case that proves it
    only narrows the blog path where it should."""
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "sources_by_name", lambda self, n: [])

    cid = oracles._resolve_handle(conn, "https://simonwillison.net")

    assert cid.startswith("blog:")


# ── the second door: `confirm(add_handles=…)` ───────────────────────────────────
#
# `add_oracle`'s preview warns about an unfiltered venue with the real count. This is the OTHER
# way the same entity gets minted, and it is the one `onboard`'s `blog` root points a new user at.


def _venue_via_confirm(conn, monkeypatch, host="https://chemrxiv.org"):
    """Confirm a venue through `add_handles`, with the OpenAlex lookup stubbed."""
    monkeypatch.setattr(oracles, "_openalex_root",
                        lambda ref: {"openalex_id": "S4393918830", "name": "ChemRxiv",
                                     "works": 63565})
    return oracles.confirm(conn, add_handles=[host])


def test_confirming_a_venue_by_handle_says_it_is_a_venue(conn, monkeypatch):
    """⚠️ THE DEFECT. `confirm` returned {canonical_id, name, handle, source} for a preprint
    repository and for a personal blog alike, so nothing distinguished a root that pulls in full
    from one where "no filter" means the newest 3%. The user reads a confirmed Oracle either way.
    """
    out = _venue_via_confirm(conn, monkeypatch)

    entry = out["confirmed"][0]
    assert entry["canonical_id"] == "openalex:S4393918830"
    assert entry["needs_topics"] is True
    assert "add_oracle" in entry["note"], "the note must name the door that has the counts"


def test_a_person_confirmed_by_handle_is_not_warned(conn, monkeypatch):
    """The warning is about the VENUE default being wrong, not about scholar roots generally. An
    author pull in full is the ruling — 928 works sits well inside `MAX_WORKS_PER_PULL`."""
    monkeypatch.setattr(oracles, "_openalex_root", lambda ref: None)
    monkeypatch.setattr(oracles, "_scholar_root", lambda ref: ("orcid.org/", "0000-0002-4027-364X"))
    monkeypatch.setattr(oracles, "_mint_orcid_entity",
                        lambda c, *a: schema.upsert_entity(c, "openalex:A5043841592",
                                                           name="F. Arnold")
                        or "openalex:A5043841592")

    out = oracles.confirm(conn, add_handles=["https://orcid.org/0000-0002-4027-364X"])

    assert out["confirmed"][0]["canonical_id"] == "openalex:A5043841592"
    assert "needs_topics" not in out["confirmed"][0]


def test_a_venue_already_narrowed_is_not_warned_again(conn, monkeypatch):
    """The warning asks for a choice. Once the choice exists, repeating it would train the host to
    ignore the one case that matters — and a re-confirm is idempotent, not a new decision."""
    _venue_via_confirm(conn, monkeypatch)
    # `set_topic_filter` writes nothing for a pair that does not exist yet — documented, and the
    # same order `record_pull` requires. A real venue's pair is seeded by its first ingest.
    st.seed_from_entities(conn, ["openalex:S4393918830"])
    st.set_topic_filter(conn, "openalex:S4393918830", "S4393918830", ["T10404"])

    out = _venue_via_confirm(conn, monkeypatch)

    assert "needs_topics" not in out["confirmed"][0]


def test_an_unreadable_registry_still_warns(conn, monkeypatch):
    """Fail-safe direction. Warning twice costs a sentence; staying silent costs 97% of a venue."""
    monkeypatch.setattr(st, "topic_filter_for",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    out = _venue_via_confirm(conn, monkeypatch)

    assert out["confirmed"][0]["needs_topics"] is True
