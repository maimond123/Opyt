"""Frontier stage 4 — SURFACE, proven offline.

Stage 4 is the only component in the rail allowed to hide anything, so the contract under test is
mostly about what it REFUSES to hide:

  • EVERY TERM DEMOTES, NOTHING GATES. A candidate with the worst possible value of every term is
    still returned. A pre-filter caps what can ever surface, and you cannot notice a thing you were
    never shown.
  • A DISMISSED CANDIDATE STILL COMES BACK, labelled and ranked last. Constraint 6: show what was
    decided, never let it vanish.
  • DEMAND COUNTS NON-VOTABLE CLAIMS. The trap the 2026-08-12 generator registry sets — `votable`
    is a liveness fact about verdicts, not a statement about whether a claim is evidence.
  • THE EVENT LOG APPENDS. Two shows are two facts; the second must not overwrite the first.
  • SUBSTANCE IS NORMALIZED WITHIN A SOURCE, so a zero-star repo is not pinned below every paper
    by a scale mismatch that has nothing to do with quality.
  • THE ORDER IS TOTAL, so a paginated read cannot silently skip or repeat a row.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import frontier_queries as fq
from pipeline.kb import frontier_surface as fs
from pipeline.kb import schema

_NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


_KIND_OF = {"arxiv": "paper", "openalex": "paper", "github": "repo"}


def _cand(conn, cid, *, source="arxiv", published="2026-08-11", summary="s" * 1200,
          payload=None, title=None, status="new", kind=None):
    conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, kind, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, source, kind or _KIND_OF.get(source, source),
         title or f"title {cid}", f"https://example/{cid}", published, summary,
         json.dumps(payload or {}), status, "2026-08-11", "2026-08-11"))
    conn.commit()
    return cid


def _link(conn, cid, text, *, generator="bookmark-reader", votable=True):
    """Attach a candidate to a standing query claimed by `generator`."""
    fq.upsert_queries(conn, [{"text": text, "target_sources": ["arxiv"],
                              "atom_ids": ["x:1"], "rationale": "r"}],
                      generator=generator, votable=votable)
    qid = fq.query_id_for(fq.normalize(text))
    conn.execute("INSERT OR IGNORE INTO frontier_candidate_queries "
                 "(candidate_id, query_id, found_at) VALUES (?,?,?)", (cid, qid, "2026-08-11"))
    conn.commit()
    return qid


def _ids(rows):
    return [r["candidate_id"] for r in rows]


# ── Nothing gates ───────────────────────────────────────────────────────────────
def test_worst_possible_candidate_is_still_returned(conn):
    """Old, no substance, found by one query, already shown five times, and dismissed. Every term
    at its worst. It ranks last and it is STILL THERE — the whole license of this stage."""
    _cand(conn, "arxiv:good", published="2026-08-12", summary="s" * 1500)
    _cand(conn, "arxiv:worst", published="2020-01-01", summary="")
    fs.record_shown(conn, ["arxiv:worst"] * 5)
    fs.record_dismissed(conn, ["arxiv:worst"])

    rows = fs.rank_candidates(conn, now=_NOW)
    assert _ids(rows) == ["arxiv:good", "arxiv:worst"]
    assert rows[-1]["state"] == "dismissed"


def test_dismissed_is_returned_marked_and_ranked_below_every_live_row(conn):
    """Dismissal is a TIER, not a big negative number: 'below everything live' must hold even for
    a dismissed candidate that would otherwise outscore the rest."""
    _cand(conn, "arxiv:dismissed-but-strong", published="2026-08-12", summary="s" * 2000)
    _link(conn, "arxiv:dismissed-but-strong", "q one")
    _link(conn, "arxiv:dismissed-but-strong", "q two")       # agreement=2, the loudest term
    _cand(conn, "arxiv:weak", published="2026-01-01", summary="")

    fs.record_dismissed(conn, ["arxiv:dismissed-but-strong"])
    rows = fs.rank_candidates(conn, now=_NOW)

    assert _ids(rows) == ["arxiv:weak", "arxiv:dismissed-but-strong"]
    assert rows[-1]["dismissed"] is True
    assert rows[-1]["state"] == "dismissed"


def test_include_dismissed_false_is_an_explicit_opt_out_only(conn):
    _cand(conn, "arxiv:a")
    _cand(conn, "arxiv:b")
    fs.record_dismissed(conn, ["arxiv:b"])

    assert _ids(fs.rank_candidates(conn, now=_NOW)) == ["arxiv:a", "arxiv:b"]
    assert _ids(fs.rank_candidates(conn, include_dismissed=False, now=_NOW)) == ["arxiv:a"]


def test_being_shown_demotes_but_never_removes(conn):
    _cand(conn, "arxiv:shown", published="2026-08-12", summary="s" * 1500)
    _cand(conn, "arxiv:fresh", published="2026-08-12", summary="s" * 1500)
    before = _ids(fs.rank_candidates(conn, now=_NOW))

    fs.record_shown(conn, ["arxiv:shown"], surface="frontier")
    after = fs.rank_candidates(conn, now=_NOW)

    assert set(before) == set(_ids(after))                    # nothing left the list
    assert _ids(after)[-1] == "arxiv:shown"                   # it only moved down
    assert after[-1]["state"] == "seen"


def test_a_stage_three_verdict_is_shown_as_state_not_used_as_a_filter(conn):
    """`frontier_candidates.status` is stage 3's, and stage 3 admitting something must not delete
    it from the surface — constraint 6 again."""
    _cand(conn, "arxiv:admitted", status="admitted")
    _cand(conn, "arxiv:rejected", status="rejected")
    rows = {r["candidate_id"]: r for r in fs.rank_candidates(conn, now=_NOW)}
    assert set(rows) == {"arxiv:admitted", "arxiv:rejected"}
    assert rows["arxiv:admitted"]["state"] == "admitted"
    assert rows["arxiv:rejected"]["state"] == "rejected"


# ── The terms ───────────────────────────────────────────────────────────────────
def test_agreement_and_demand_vanish_at_one_so_the_ranking_degrades_to_the_rest(conn):
    """On the only data that exists every candidate has exactly one query and one generator. Both
    bonuses must be worth ZERO there, or the sort key is the same constant on every row."""
    assert fs._bonus(1, fs.W_AGREEMENT) == 0.0
    assert fs._bonus(0, fs.W_DEMAND) == 0.0            # absent reads as one, not as negative
    assert fs._bonus(2, fs.W_AGREEMENT) == pytest.approx(fs.W_AGREEMENT)

    _cand(conn, "arxiv:old", published="2026-06-01")
    _cand(conn, "arxiv:new", published="2026-08-12")
    _link(conn, "arxiv:old", "q one")
    _link(conn, "arxiv:new", "q two")
    assert _ids(fs.rank_candidates(conn, now=_NOW)) == ["arxiv:new", "arxiv:old"]


def test_agreement_promotes_a_multi_query_hit(conn):
    """The schema's stated reason for `frontier_candidate_queries` being a table at all."""
    _cand(conn, "arxiv:one-query", published="2026-08-12", summary="s" * 2000)
    _cand(conn, "arxiv:three-queries", published="2026-07-01", summary="s" * 800)
    _link(conn, "arxiv:one-query", "solo query")
    for t in ("query a", "query b", "query c"):
        _link(conn, "arxiv:three-queries", t)

    rows = fs.rank_candidates(conn, now=_NOW)
    assert _ids(rows)[0] == "arxiv:three-queries"
    assert rows[0]["n_queries"] == 3
    assert "3 standing queries found it" in rows[0]["why"]


def test_demand_counts_claims_from_a_NON_VOTABLE_generator(conn):
    """THE REGISTRY TRAP. `votable=0` means "this channel can never change its mind about this
    query", which exists so a frozen counter cannot pin a query's speed. It does NOT mean the
    claim is not evidence: a write-once generator can never vote, yet its claim is still proof
    that a region of the KB asked for this thread. Filtering on `votable` here would silently
    discard exactly the claims that make demand worth measuring.

    The non-votable claim is CONSTRUCTED here rather than taken from a real rail. `sitting:*` was
    the live example until 2026-08-16, when the sitting reader was taught verdicts (D11) and
    flipped to votable; the generator name below is now just a plausible label. What is pinned is
    the ranking rule, which has to hold for whichever generator is write-once next."""
    _cand(conn, "arxiv:wanted")
    _link(conn, "arxiv:wanted", "shared thread", generator="bookmark-reader", votable=True)
    _link(conn, "arxiv:wanted", "shared thread", generator="sitting:mlx", votable=False)

    # Precondition: the registry really does hold that generator as non-votable.
    assert conn.execute("SELECT votable FROM frontier_generators WHERE generator='sitting:mlx'"
                        ).fetchone()[0] == 0

    row = fs.rank_candidates(conn, now=_NOW)[0]
    assert row["n_generators"] == 2, "a non-votable claim is still demand"
    assert row["score"] > 0
    assert "2 regions of your KB asked for it" in row["why"]


def test_a_retired_query_still_counts_what_it_already_found(conn):
    """Retiring a query stops it RUNNING; it does not un-find what it found."""
    _cand(conn, "arxiv:found-by-retired")
    _link(conn, "arxiv:found-by-retired", "live thread")
    _link(conn, "arxiv:found-by-retired", "dead thread")
    fq.retire_query(conn, "dead thread")

    row = fs.rank_candidates(conn, now=_NOW)[0]
    assert row["n_queries"] == 2


def test_substance_is_normalized_within_a_source(conn):
    """arXiv summaries are real abstracts (717-1900 chars measured); GitHub descriptions average
    197. On one scale every repo sits below every paper forever — a filter wearing a re-rank's
    clothes. A repo that is strong FOR A REPO must be able to outrank a weak paper."""
    _cand(conn, "repo:strong", source="github", published="2026-08-12",
          summary="short one-liner", payload={"stars": 9000})
    _cand(conn, "repo:weak", source="github", published="2026-08-12",
          summary="short one-liner", payload={"stars": 0})
    _cand(conn, "arxiv:strong", source="arxiv", published="2026-08-12", summary="s" * 1800)
    _cand(conn, "arxiv:weak", source="arxiv", published="2026-08-12", summary="s" * 100)

    rows = {r["candidate_id"]: r for r in fs.rank_candidates(conn, now=_NOW)}
    assert rows["repo:strong"]["substance"] == pytest.approx(1.0)
    assert rows["arxiv:strong"]["substance"] == pytest.approx(1.0)
    assert rows["repo:strong"]["score"] > rows["arxiv:weak"]["score"]


def test_a_saturating_signal_does_not_collapse_the_top_of_its_own_source(conn):
    """REGRESSION, caught on the live baseline 2026-08-12. `_raw_substance` used to clamp an arXiv
    abstract at 1200 characters. 21 of the 26 baseline abstracts ran past that, so they all tied at
    the ceiling, midrank dropped the entire block to 0.60, and no paper could reach the substance
    a repo could — which swept EVERY arXiv row out of the top 20 of 111. The percentile reads only
    the order, so a ceiling can never help and can silently bury a whole source."""
    for i in range(10):                       # nine well past any plausible "full abstract" cap
        _cand(conn, f"arxiv:{i}", source="arxiv", summary="s" * (1500 + i * 100))
    _cand(conn, "arxiv:stub", source="arxiv", summary="s" * 80)

    rows = {r["candidate_id"]: r for r in fs.rank_candidates(conn, now=_NOW)}
    assert rows["arxiv:9"]["substance"] == pytest.approx(1.0), "the longest must reach the top"
    assert rows["arxiv:stub"]["substance"] == pytest.approx(0.0)
    assert len({r["substance"] for r in rows.values()}) == 11, "the cap fused distinct rows"


def test_neither_source_is_shut_out_of_the_head_of_a_mixed_queue(conn):
    """The property the regression above broke. With both sources spread over their own
    distributions, a mixed queue's head must not be one source by construction."""
    for i in range(30):
        _cand(conn, f"repo:{i:02d}", source="github", published="2026-08-10",
              summary="one-liner", payload={"stars": i * 40})
    for i in range(15):
        _cand(conn, f"arxiv:{i:02d}", source="arxiv", published="2026-08-10",
              summary="s" * (700 + i * 90))

    head = [r["source"] for r in fs.rank_candidates(conn, limit=10, now=_NOW)]
    assert "arxiv" in head and "github" in head, f"one source swept the head: {head}"


def test_a_lone_row_of_a_source_gets_the_neutral_substance(conn):
    """One observation has no distribution. Scoring it 0.0 or 1.0 is an opinion the data does not
    support, and 0.0 would silently bury the first candidate any new adapter ever returns."""
    _cand(conn, "repo:only", source="github", payload={"stars": 0})
    _cand(conn, "arxiv:a", source="arxiv")
    _cand(conn, "arxiv:b", source="arxiv", summary="tiny")
    rows = {r["candidate_id"]: r for r in fs.rank_candidates(conn, now=_NOW)}
    assert rows["repo:only"]["substance"] == pytest.approx(0.5)


def test_an_unreadable_date_is_neutral_not_ancient(conn):
    """Unknown must not read as old — that demotes by parser accident rather than by evidence."""
    assert fs._recency(None, now=_NOW) == pytest.approx(0.5)
    assert fs._recency("not a date", now=_NOW) == pytest.approx(0.5)
    # A published date is a DAY and `now` is an instant, so "today" already carries part of a
    # day's age. That is a rounding artefact of comparing the two, not a demotion.
    assert fs._recency("2026-08-12", now=_NOW) == pytest.approx(1.0, abs=0.02)
    assert fs._recency("2026-07-13", now=_NOW) == pytest.approx(0.5, abs=0.02)
    assert fs._recency("2019-01-01", now=_NOW) > 0.0     # decays toward zero, never to it


def test_an_unknown_source_is_not_born_demoted(conn):
    _cand(conn, "zenodo:1", source="zenodo", summary="")
    _cand(conn, "zenodo:2", source="zenodo", summary="")
    rows = fs.rank_candidates(conn, now=_NOW)
    assert all(r["score"] > 0 for r in rows)


# ── Order and pagination ────────────────────────────────────────────────────────
def test_the_order_is_total_and_stable_across_calls(conn):
    """Identical rows in every term. Without `candidate_id` as the last sort key their order is
    whatever SQLite felt like, and a paginated read would skip and repeat rows between calls."""
    for i in range(12):
        _cand(conn, f"arxiv:tie-{i:02d}", published="2026-08-12", summary="s" * 1200)

    first = _ids(fs.rank_candidates(conn, now=_NOW))
    second = _ids(fs.rank_candidates(conn, now=_NOW))
    assert first == second == sorted(first)

    page = _ids(fs.rank_candidates(conn, limit=5, now=_NOW))
    assert page == first[:5]


def test_limit_truncates_without_touching_the_store(conn):
    for i in range(6):
        _cand(conn, f"arxiv:{i}")
    assert len(fs.rank_candidates(conn, limit=2, now=_NOW)) == 2
    assert len(fs.rank_candidates(conn, now=_NOW)) == 6      # nothing was consumed


def test_ranking_an_empty_store_is_empty_not_an_error(conn):
    assert fs.rank_candidates(conn, now=_NOW) == []


def test_a_missing_table_degrades_to_empty(tmp_path):
    """Fail-safe: a read-only or pre-stage-2 store must yield nothing, never raise."""
    bare = sqlite3.connect(":memory:")
    bare.row_factory = sqlite3.Row
    assert fs.rank_candidates(bare, now=_NOW) == []
    bare.close()


# ── The event log ───────────────────────────────────────────────────────────────
def test_the_event_log_appends_rather_than_overwrites(conn):
    _cand(conn, "arxiv:a")
    fs.record_shown(conn, ["arxiv:a"], surface="frontier")
    fs.record_shown(conn, ["arxiv:a"], surface="search")

    rows = conn.execute("SELECT event, surface FROM frontier_candidate_events "
                        "WHERE candidate_id='arxiv:a' ORDER BY event_id").fetchall()
    assert [r["surface"] for r in rows] == ["frontier", "search"]
    assert fs.rank_candidates(conn, now=_NOW)[0]["shown_n"] == 2


def test_shown_and_dismissed_coexist_on_one_candidate(conn):
    """The sequence is the point — a dismissal does not erase the record of having been shown."""
    _cand(conn, "arxiv:a")
    fs.record_shown(conn, ["arxiv:a"])
    fs.record_dismissed(conn, ["arxiv:a"])
    events = [r["event"] for r in conn.execute(
        "SELECT event FROM frontier_candidate_events ORDER BY event_id")]
    assert events == ["shown", "dismissed"]


def test_record_event_dedups_within_one_call_and_ignores_blanks(conn):
    _cand(conn, "arxiv:a")
    assert fs.record_shown(conn, ["arxiv:a", "arxiv:a", "", None]) == 1


# ── The notice ──────────────────────────────────────────────────────────────────
def test_notice_is_none_on_an_empty_store(conn):
    assert fs.notice(conn, now=_NOW) is None


def test_notice_is_none_once_everything_has_been_shown(conn):
    """Silence must be genuinely zero-footprint, or the push surface earns being ignored."""
    _cand(conn, "arxiv:a")
    assert fs.notice(conn, now=_NOW) is not None
    fs.record_shown(conn, ["arxiv:a"])
    assert fs.notice(conn, now=_NOW) is None


def test_notice_counts_unshown_and_names_the_strongest(conn):
    _cand(conn, "arxiv:weak", published="2026-05-01", summary="")
    _cand(conn, "arxiv:strong", published="2026-08-12", summary="s" * 1800, title="The Strong One")
    n = fs.notice(conn, now=_NOW)
    assert n["unshown"] == 2 and n["total"] == 2
    assert n["top"]["candidate_id"] == "arxiv:strong"
    assert "The Strong One" in n["message"]
    assert "summary" not in n["top"], "a notice is a count, never the digest"


def test_notice_ignores_dismissed_rows_in_its_count(conn):
    _cand(conn, "arxiv:a")
    _cand(conn, "arxiv:b")
    fs.record_dismissed(conn, ["arxiv:b"])
    assert fs.notice(conn, now=_NOW)["unshown"] == 1


def test_notice_never_raises_through_its_carrier(conn, monkeypatch):
    """It rides on `search`'s response. A DB hiccup here must cost the notice, never the
    search — the fail-safe invariant."""
    _cand(conn, "arxiv:a")          # get past the cheap COUNT so the guarded pass really runs
    monkeypatch.setattr(fs, "rank_candidates",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fs.notice(conn, now=_NOW) is None


def test_the_quiet_case_never_ranks_anything(conn, monkeypatch):
    """This carrier fires on an ordinary search, so 'nothing new' must not pay for a full ranking
    pass over a table that only grows."""
    _cand(conn, "arxiv:a")
    fs.record_shown(conn, ["arxiv:a"])
    monkeypatch.setattr(fs, "rank_candidates",
                        lambda *a, **k: pytest.fail("ranked on the quiet path"))
    assert fs.notice(conn, now=_NOW) is None


def test_substance_is_read_off_the_kind_not_the_finder():
    """KIND says what the signal IS; SOURCE says what it is comparable to. A second paper finder
    keyed on source would fall to the 0.0 default, lose the substance term entirely, and a third
    would be a third name in the same arm — the duplication the finder/minter split removed from
    stage 3, surviving in one function."""
    assert fs._raw_substance("paper", "s" * 400, {}) == 400.0        # any paper finder
    assert fs._raw_substance("repo", None, {"stars": 12}) == 12.0
    assert fs._raw_substance(None, "s" * 400, {}) == 0.0             # unknown → neutral percentile


def test_two_paper_finders_are_ranked_against_their_own_distributions(conn):
    """Substance stays normalized WITHIN a source even though it is READ per kind. Abstract length
    does not mean the same thing across providers — OpenAlex reconstructs abstracts from an
    inverted index, arXiv ships them verbatim — so pooling them would rank one provider's
    formatting as the other's depth."""
    for i in range(4):
        _cand(conn, f"arxiv:{i}", source="arxiv", summary="s" * (100 + i))
        _cand(conn, f"openalex:W{i}", source="openalex", summary="s" * (5000 + i))

    ranked = {r["candidate_id"]: r["substance"] for r in fs.rank_candidates(conn, now=_NOW)}
    assert ranked["arxiv:3"] == 1.0 and ranked["openalex:W3"] == 1.0   # each tops its own group
    assert ranked["arxiv:0"] == 0.0 and ranked["openalex:W0"] == 0.0


# ── one artifact, one card ──────────────────────────────────────────────────────
# A finder's id is a fact about the FINDER, not about the thing it found, so one artifact reaches
# the queue under several `candidate_id`s routinely. Measured over 68 live OpenAlex results on
# 2026-08-26: 12 groups covering 24 of the 68 works. The three shapes are pinned below.
_AUTH = {"authors": ["Vishisht Choudhary"]}


def _twin(conn, cid, **kw):
    """Two staged rows that are the same artifact: same title, author and date, different ids."""
    kw.setdefault("published", "2026-07-29")
    return _cand(conn, cid, source="openalex", kind="paper", title="Detecting An AI Agent",
                 payload=dict(_AUTH), **kw)


def test_the_same_artifact_under_two_candidate_ids_is_one_card(conn):
    """The arXiv twin: OpenAlex returns one preprint twice, once with `doi: null` and an `/abs/`
    landing page and once with its 10.48550 DOI. 7 of the 12 measured groups have this shape."""
    _twin(conn, "openalex:W1")
    _twin(conn, "openalex:W2")
    _link(conn, "openalex:W1", "agents")
    _link(conn, "openalex:W2", "agents")

    ranked = fs.rank_candidates(conn)
    assert len(ranked) == 1
    assert ranked[0]["duplicate_of"] == [c for c in ("openalex:W1", "openalex:W2")
                                         if c != ranked[0]["candidate_id"]]
    assert any("staged 2x" in w for w in ranked[0]["why"])


def test_the_collapse_merges_evidence_as_a_max_never_a_sum(conn):
    """A merge, not a filter — but a SUM would double-count one standing query that found both
    forms of the artifact. Max can only under-state, which is the safe direction for a ranker."""
    _twin(conn, "openalex:W1")
    _twin(conn, "openalex:W2")
    _link(conn, "openalex:W1", "agents")
    _link(conn, "openalex:W2", "agents")          # the SAME query found both forms

    ranked = fs.rank_candidates(conn)
    assert len(ranked) == 1
    assert ranked[0]["n_queries"] == 1            # 1, not 2 — one query found one artifact


def test_a_row_with_no_title_is_never_merged(conn):
    """The mass-collapse guard. A key that groups on ABSENCE would fold every untitled row into
    one card, which is how a collapse becomes a silent delete. Unkeyable rows stay separate."""
    _cand(conn, "openalex:A", source="openalex", kind="paper", title="", published="2026-07-29")
    _cand(conn, "openalex:B", source="openalex", kind="paper", title="", published="2026-07-29")

    assert len(fs.rank_candidates(conn)) == 2


def test_a_shared_title_with_different_dates_is_two_artifacts(conn):
    """Date is required, not decorative: an identically-titled weekly report series is real, and
    collapsing it would hide every issue but one."""
    _twin(conn, "openalex:W1")
    _twin(conn, "openalex:W2", published="2026-08-05")

    assert len(fs.rank_candidates(conn)) == 2


def test_a_materialized_twin_makes_the_card_materialized(conn):
    """A group can straddle an admit run, so one row can be materialized while its twin is still
    `new`. The artifact is in the KB either way and the card must not report it as unseen work."""
    _twin(conn, "openalex:W1", status="new")
    _twin(conn, "openalex:W2", status="materialized")

    ranked = fs.rank_candidates(conn)
    assert len(ranked) == 1
    assert ranked[0]["state"] == "materialized"


def test_the_stars_line_survives_a_null_kind(conn):
    """`_why` reads the PAYLOAD KEY, not `source` and not `kind`, and the NULL row is why.

    Switching the old `source == "github"` test to `kind == "repo"` would have read correctly and
    silently dropped the line for every row staged before the finder/minter split, whose `kind` is
    NULL by decision — `init_kb_schema` adds the column with NO backfill, because a converged
    migration re-running on every connect is the measured cause of the live lock contention. The
    key's presence is the whole question, so neither dispatch was ever load-bearing here.
    """
    # Inserted directly: `_cand` defaults a missing kind through `_KIND_OF`, which is exactly the
    # source->kind coincidence this row exists to NOT have.
    conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, kind, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES ('repo:legacy','github',NULL,'owner/repo','https://example/legacy',
                   '2026-08-11','one-liner','{"stars": 4200}','new','2026-08-11','2026-08-11')""")
    conn.commit()
    _link(conn, "repo:legacy", "repos")

    (card,) = fs.rank_candidates(conn)
    assert card["kind"] is None
    assert "4200 stars" in card["why"]
