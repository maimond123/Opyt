"""`run_kb_search`'s response envelope — `{hits, notices, insights, trace, frontier_atoms}`.

A bare list has no header, so there would be nowhere to report a fact about the result (which
arms ran, what units `score` is in, what a filter cost you). These pin the keys, and
specifically pin the silent behaviours that need a reporting channel.

`frontier_atoms`: a default call ranks its full k over HUMAN_ATTESTED atoms and answers the
keyword-crawl population beside that list rather than inside it.

The rule under test throughout: `notices` are sentences (repeat them to the user),
`insights`/`trace` are values (reason over them, never recite).
"""
from __future__ import annotations

from pipeline.retrieval import local
from pipeline.retrieval import reads as kb_entry
from pipeline.curation import imports
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.retrieval import cards
from mcp_server.rider_rules import ONCE_PER_CONVERSATION
from tests import mcp_session
from pipeline.oracles import roster
from pipeline.oracles import freshness

A = "github:root/agentkit"
B = "github:stranger/agents"
C = "x:1"
P = "paper:2401.00001"


def _add(conn, atom_id, source_type, what_kind, who_id, topics, snapshot,
         *, entry_mode="oracle-footprint", when_ts="2024-05-01", body_state=None):
    raw_hash = snapshot_hash(snapshot)
    payload = {"source_tags": topics}
    if body_state is not None:
        payload["body_state"] = body_state
    store_atom(conn, dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts=when_ts, when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_hash=raw_hash,
        description=f"{atom_id} card", payload=payload, entry_mode=entry_mode,
    ), snapshot)


def _corpus(conn):
    _add(conn, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents",
         entry_mode="user-saved", when_ts="2024-05-01", body_state="complete")
    _add(conn, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", entry_mode="oracle-footprint", when_ts="2024-07-09",
         body_state="partial")
    _add(conn, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems", entry_mode="user-saved")
    _add(conn, P, "paper", "artifact", "scholar:1", ["ai-agents"],
         "a paper about an agent framework and its library", entry_mode="oracle-footprint")


# ── the shape ────────────────────────────────────────────────────────────────────

def test_an_ordinary_query_returns_all_five_keys(kb_home):
    """`frontier_atoms` is present on every default call —
    including one that found no frontier atoms, so "none matched" and "this build has no such
    list" cannot look the same. Scoping the call to a population removes it: the whole answer
    is then that population and a second list would be the same list."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", k=8)
    assert set(out) == {"hits", "notices", "insights", "trace", "frontier_atoms"}
    assert out["frontier_atoms"]["hits"] == []          # the corpus is all human-attested

    scoped = kb_entry.run_kb_search("framework", entry_mode="frontier", k=8)
    assert set(scoped) == {"hits", "notices", "insights", "trace"}


def test_hits_are_exactly_what_the_bare_list_used_to_hold(kb_home):
    """The envelope wraps; it must not move ranking or hit content. Same ids, same order, same
    card keys as the underlying search — everything new lives outside `hits`."""
    conn = schema.connect(); _corpus(conn)
    from pipeline.retrieval.engine import search_atoms
    raw = search_atoms(conn, "framework", k=8)
    conn.close()

    hits = kb_entry.run_kb_search("framework", k=8)["hits"]
    assert [h["atom_id"] for h in hits] == [h.atom_id for h in raw.hits]
    assert {"citation_id", "atom_id", "snippet", "source_url", "score", "body_state",
            "payload"} <= set(hits[0])


def test_a_healthy_query_reports_no_notices(kb_home):
    """THE contract for `notices`: emptiness is meaningful. If a clean query carried chatter,
    a host would learn to skip the field and would then skip the real ones too."""
    conn = schema.connect(); _corpus(conn); conn.close()
    assert kb_entry.run_kb_search("framework", k=8)["notices"] == []


# ── a tag that slugifies to nothing, asserted through the envelope ────────────────────────────────

def test_a_tag_that_normalizes_away_matches_nothing_and_says_so(kb_home):
    """`slugify("+++")` is empty, so it could leave the pre-filter with no clauses at all
    and return the whole store under a filter the caller believed was applied. Two assertions,
    because they are two separate failures: returning everything is the bug, and staying quiet
    about the dropped tag is what made it invisible."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["+++"], k=8)

    assert out["hits"] == []                                # not the whole store
    codes = [n["code"] for n in out["notices"]]
    assert "tags_normalized" in codes
    assert "+++" in next(n for n in out["notices"] if n["code"] == "tags_normalized")["message"]


def test_a_mangled_tag_is_reported_even_though_it_still_matches(kb_home):
    """The half the pre-filter fix cannot catch. `"C++"` slugifies to `"c"` — a valid, entirely
    different filter — so the query runs and succeeds while answering a question nobody asked.
    A silent substitution has no symptom; only the notice makes it visible."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["C++"], k=8)

    note = next(n for n in out["notices"] if n["code"] == "tags_normalized")
    assert note["changed"] == [["C++", "c"]]
    assert out["trace"]["filters"]["tags"] == ["c"]          # filters as applied, not as asked


def test_a_cost_sentence_names_a_filter_value_only_when_there_is_one(kb_home):
    """A notice is read out to the user verbatim, so it has to survive the degenerate case. A
    filter that normalized away has an applied value of `[]`, and "Dropping tags=[] would
    return 4" is noise — the sentence explaining WHY it is empty is a separate notice."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["+++"], k=8)

    msg = next(n for n in out["notices"] if n["code"] == "filters_matched_nothing")["message"]
    assert "tags=[]" not in msg
    assert "Dropping tags would return 4." in msg


# ── empty results have four different causes ─────────────────────────────────────

def test_an_empty_store_says_so_rather_than_returning_a_bare_empty(kb_home):
    schema.connect().close()                                 # schema only, zero atoms
    out = kb_entry.run_kb_search("framework", k=8)
    assert out["hits"] == []
    assert [n["code"] for n in out["notices"]] == ["store_empty"]
    # A dead end is worse than no notice: whoever reads this has an empty store and no idea what
    # to do about it, and the actual blocker (two unacquired API keys) is nowhere in the sentence.
    note = out["notices"][0]
    assert note["next_tool"] == "onboard"
    assert "onboard" in note["message"]


def test_filters_that_match_nothing_report_what_the_filter_cost(kb_home):
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["ai-agents"], source_type="substack", k=8)

    note = next(n for n in out["notices"] if n["code"] == "filters_matched_nothing")
    assert out["insights"]["filter_cost"]["source_type"] == 3     # A, B, P are ai-agents
    assert "source_type" in note["message"]
    assert out["trace"]["candidates"] == 0 and out["trace"]["fts_query"] is None  # nothing ran


def test_an_unresolvable_handle_is_distinguishable_from_a_quiet_person(kb_home):
    """An unresolvable handle and a person who wrote nothing both give `[]`; one call tells them apart."""
    conn = schema.connect(); _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")
    conn.close()

    untracked = kb_entry.run_kb_search("framework", who="@nobody", k=8)
    assert [n["code"] for n in untracked["notices"]] == ["who_unresolved"]
    assert "@nobody" in untracked["notices"][0]["message"]
    # This is the local-only tool's only channel back to a host that has web search and this
    # one doesn't — a miss must point the host at it, not just report the miss.
    assert "web-search" in untracked["notices"][0]["message"]
    assert untracked["insights"]["resolved_who"] == []

    quiet = kb_entry.run_kb_search("dashboard", who="root", k=8)
    assert quiet["hits"] == []                                   # they wrote nothing on it...
    assert quiet["insights"]["resolved_who"][0]["name"] == "Root McRepo"   # ...but we have them
    assert "who_unresolved" not in [n["code"] for n in quiet["notices"]]


def test_a_handle_matching_two_people_warns_that_results_mix_them(kb_home):
    """`resolve_who` returns one entry per cluster. Two clusters means the hits are two
    different people's work presented as one person's — worth a sentence, not a value."""
    conn = schema.connect(); _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Root One")
    schema.upsert_entity(conn, "substack:root", name="Root Two")
    conn.close()

    out = kb_entry.run_kb_search("framework", who="root", k=8)
    note = next(n for n in out["notices"] if n["code"] == "who_multiple")
    assert len(note["canonical_ids"]) == 2
    assert "Root One" in note["message"] and "Root Two" in note["message"]


# ── filter_cost is self-pruning, which is what replaces a threshold ───────────────

def test_filter_cost_is_empty_when_every_filter_is_load_bearing(kb_home):
    """No "is this result thin enough to explain" threshold exists anywhere, because a
    well-targeted query prices out at `{}` by construction. `what_kind="artifact"` excludes
    only C, which the tag filter already excluded — so dropping it would gain nothing."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["ai-agents"], what_kind="artifact", k=8)
    assert out["insights"]["filter_cost"] == {}


def test_filter_cost_is_reported_even_when_the_query_looks_healthy(kb_home):
    """The dangerous case, and the reason this is computed on every search rather than only on
    thin ones: a filter narrows the result, the hits still look clean, and nothing anywhere
    says what was excluded. A silent narrowing has no symptom to trigger on."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", source_type="github", k=8)

    assert len(out["hits"]) == 2                       # a perfectly healthy-looking result...
    assert out["insights"]["filter_cost"]["source_type"] == 2   # ...that excluded C and P


# ── trace: what the engine did ───────────────────────────────────────────────────

def test_the_rewritten_keyword_query_is_visible(kb_home):
    """A BM25 hit needs ONE token, not the phrase — `fts_query` ORs them. That rewrite
    explains hits that otherwise look irrelevant, and nobody could see it."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework", k=8)
    assert out["trace"]["fts_query"] == '"agent" OR "framework"'


def test_truncation_reports_the_score_at_the_boundary(kb_home):
    """The instrument for "did top-k cut off good answers". `candidates` is a filter fact — how
    many cleared the pre-filter, most possibly irrelevant. The two scores at the cut are the
    relevance fact: near-identical means the cut landed mid-cluster, a large drop means k was
    already the right number. Reported, never judged — no threshold here."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", k=2)

    assert len(out["hits"]) == 2
    assert out["trace"]["ranked"] == 3                 # A, B, P all matched
    cut = out["trace"]["cutoff"]
    assert cut["last_returned"] == out["hits"][-1]["score"]
    assert cut["next_best"] < cut["last_returned"]     # rank 3's score, the one you didn't get
    assert [n["code"] for n in out["notices"]].count("results_truncated") == 1


def test_cutoff_is_absent_when_nothing_was_cut(kb_home):
    """Absent, not zeroed. There is no boundary to describe, and a 0.0 would read as one."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", k=8)
    assert "cutoff" not in out["trace"]
    assert [n["code"] for n in out["notices"]] == []


def test_candidates_separates_unfiltered_from_filtered_to_nothing(kb_home):
    """`None` (no filter ran, whole store searched) and `0` (a filter ran, nothing survived)
    are opposite facts, so they must not share a value.

    `None` is not reachable through `run_kb_search`: a default call always scopes to
    HUMAN_ATTESTED, so a count is always a real count. The distinction still has to hold one layer down, where an unscoped search lives."""
    conn = schema.connect(); _corpus(conn)
    from pipeline.retrieval.engine import search_atoms
    assert search_atoms(conn, "framework").candidates is None
    conn.close()
    assert kb_entry.run_kb_search("framework")["trace"]["candidates"] == 4
    assert kb_entry.run_kb_search("framework", source_type="youtube")["trace"]["candidates"] == 0


# ── insights: distributions over the hits, reported not judged ───────────────────

def test_insights_summarize_exactly_the_hits_they_describe(kb_home):
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", k=8)
    ins, hits = out["insights"], out["hits"]

    assert sum(ins["sources"].values()) == len(hits)
    assert ins["sources"] == {"github": 2, "paper": 1}
    assert ins["body_state"]["complete"] == 1 and ins["body_state"]["partial"] == 1
    assert ins["date_span"] == ["2024-05-01", "2024-07-09"]


def test_entry_mode_reaches_insights_from_a_column_no_query_ever_read(kb_home):
    """`atoms.entry_mode` records real provenance: 'you saved this yourself' vs 'we swept it because
    we track the author'.

    The tally is a pass-through of whatever modes appear, not a two-way split, even though the
    key name suggests one."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library rollup", k=8)
    assert out["insights"]["saved_vs_crawled"] == {"oracle-footprint": 2, "user-saved": 2}


def test_a_lopsided_result_is_a_distribution_never_a_warning(kb_home):
    """Every hit here is by one author. `insights` reports the concentration
    and the host decides whether that is alarming; `notices` must not editorialize about
    evidence quality, because that is a judgement the tool has no standing to make."""
    conn = schema.connect(); _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")
    conn.close()

    out = kb_entry.run_kb_search("autonomous", k=8)
    assert out["insights"]["authors"] == {"Root McRepo": 1}
    assert out["notices"] == []                        # a distribution, not a warning


def test_corpus_newest_is_the_whole_store_not_the_hits(kb_home):
    """A fact no hit can carry: "my newest atom is months old" is what explains a result that
    looks stale, and it is invisible from inside a list of hits."""
    conn = schema.connect(); _corpus(conn); conn.close()
    out = kb_entry.run_kb_search("framework", k=8)
    assert out["insights"]["corpus_newest"] is not None


# ── through the real server ─────────────────────────────────────────────────────

def _tools():
    """The atom tools through the real server, in this test's one MCP session — the path the
    riders take (`mcp_server/riders.py`)."""
    return mcp_session.session().tools()


# ── the freshness notice rides on `search`, gated on needs_attention ─────────
#
# Decision 4 said freshness rides on `search` AND `oracle(action="screen")`. Only `screen`
# shipped. Users go to `search`; they rarely go to `screen`, and a frozen roster has to be
# visible where people actually are.

def test_stale_oracles_are_reported_on_search(kb_home, monkeypatch):
    monkeypatch.setattr(freshness, "status_summary",
                        lambda conn: {"needs_attention": True, "stale_fraction": 0.9})
    conn = schema.connect()
    _add(conn, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = _tools()["search"]("framework", k=8)
    note = next(n for n in out["notices"] if n["code"] == "oracles_stale")
    assert "refreshed" in note["message"]


def test_fresh_oracles_add_no_notice(kb_home, monkeypatch):
    """Gated, not unconditional — unlike `screen`, which is deliberate and occasional.
    `search` is high-frequency, and a block printed on every call trains the reader to skip
    the one call where it matters."""
    monkeypatch.setattr(freshness, "status_summary",
                        lambda conn: {"needs_attention": False})
    conn = schema.connect()
    _add(conn, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = _tools()["search"]("framework", k=8)
    assert "oracles_stale" not in [n["code"] for n in out["notices"]]


def test_a_freshness_hiccup_never_breaks_a_search(kb_home, monkeypatch):
    monkeypatch.setattr(freshness, "status_summary",
                        lambda conn: (_ for _ in ()).throw(RuntimeError("db locked")))
    conn = schema.connect()
    _add(conn, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    assert _tools()["search"]("framework", k=8)["hits"]


# ── the sectioned response ────────────────────────────────────
# A frontier atom's score is not comparable to a human-attested one's — papers are
# 24x denser than X posts, and a result set should not be mostly papers. So the
# two populations are ranked apart and labeled, and no score crosses between them.

def _with_frontier(conn):
    """The `_corpus` atoms plus three frontier ones whose scores fall off a cliff after the head,
    so the block's own-top floor has a tail to remove."""
    _corpus(conn)
    _add(conn, "paper:crawl/strong", "paper", "artifact", "openalex:1", ["ai-agents"],
         "an agent framework library", entry_mode="frontier")
    _add(conn, "paper:crawl/mid", "paper", "artifact", "openalex:2", ["ai-agents"],
         "an agent framework library for tools", entry_mode="frontier")
    _add(conn, "github:crawl/weak", "github", "artifact", "github:crawler", ["crypto"],
         "a rollup proof dashboard", entry_mode="frontier")


def test_a_default_call_ranks_humans_alone_and_labels_the_crawl_beside_them(kb_home):
    """`hits` holds no frontier atom however well it scored, and the crawl
    is not dropped either — it is returned under its own key, where a host can offer it as a
    different kind of evidence rather than as more of the same."""
    conn = schema.connect(); _with_frontier(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", k=8)

    assert out["hits"], "the human-attested list must still be the answer"
    assert all(h["entry_mode"] != "frontier" for h in out["hits"])
    assert {h["atom_id"] for h in out["frontier_atoms"]["hits"]} <= {
        "paper:crawl/strong", "paper:crawl/mid", "github:crawl/weak"}
    assert out["frontier_atoms"]["candidates"] == 3          # cleared the frontier scope


def test_an_explicit_scope_returns_one_list_and_no_second_one(kb_home):
    """"More frontier" needed no new tool and no new surface — it is the same call, scoped.
    Pinned because the alternative (a separate frontier search tool) was rejected on the
    cost of host tool-selection past ~30 tools."""
    conn = schema.connect(); _with_frontier(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", entry_mode="frontier", k=20)

    assert out["hits"] and all(h["entry_mode"] == "frontier" for h in out["hits"])
    assert "frontier_atoms" not in out
    assert out["trace"]["filters"]["entry_mode"] == "frontier"   # applied, and said so


def test_a_broken_frontier_pass_costs_the_answer_nothing(kb_home, monkeypatch):
    """Fail-safe: the block is an addition to the answer; a query that
    would have worked before it existed must still work. Degrade to an empty block plus a
    notice — never a raised exception, and never a silently missing key."""
    conn = schema.connect(); _with_frontier(conn); conn.close()

    def _boom(*a, **kw):
        raise RuntimeError("frontier pass exploded")
    monkeypatch.setattr(local, "_frontier_block", _boom)
    out = kb_entry.run_kb_search("agent framework library", k=8)

    assert out["hits"]                                    # the real answer survived
    assert out["frontier_atoms"]["hits"] == []
    note = next(n for n in out["notices"] if n["code"] == "frontier_block_failed")
    assert "RuntimeError" in note["error"]


def test_a_card_says_which_list_it_came_from(kb_home):
    """`entry_mode` is user-facing policy, not just a filter column. Cards get copied out of
    their list — into a citation, a follow-up call — and one read alone must still say whether a
    person vouched for it or a keyword query found it."""
    conn = schema.connect(); _with_frontier(conn); conn.close()
    out = kb_entry.run_kb_search("agent framework library", k=8)

    assert {h["entry_mode"] for h in out["hits"]} <= set(schema.HUMAN_ATTESTED)
    assert all(h["entry_mode"] == "frontier" for h in out["frontier_atoms"]["hits"])


def test_the_search_tool_exposes_the_scope_and_keeps_the_latest_notice_apart(
        kb_home, monkeypatch):
    """`frontier_atoms` is the crawl's atoms that matched this query; `latest` counts everything
    that arrived since the user last looked, whatever the query."""
    conn = schema.connect(); _with_frontier(conn); conn.close()
    search = _tools()["search"]
    monkeypatch.setattr("pipeline.arrivals.notice.notice", lambda conn, now: {"arrived": 7})

    out = search("agent framework library", k=8)

    assert out["latest"] == {"arrived": 7, "host_note": ONCE_PER_CONVERSATION}
    assert "hits" in out["frontier_atoms"]                 # this query's crawl results
    scoped = search("agent framework library", entry_mode="frontier")
    assert "frontier_atoms" not in scoped


# ── the thin-coverage enrichment offer ──────────────────────────────────────────
#
# Why it exists: enrichment (watch a subject + survey the web on it) is not only for onboarding
# moments. The highest-intent moment is mid-use: a
# user who searches for a subject their store barely covers has just said, by asking, that they
# care about something OPYT is thin on. Root-agnostic, and better signal than any tag count.

def test_a_thin_unfiltered_search_offers_enrichment_once_per_conversation(kb_home):
    """The server keeps no memory of the conversation, so the offer rides every thin search and
    asks the host, which can see whether it already made it, to make it once."""
    conn = schema.connect(); _corpus(conn); conn.close()
    search = _tools()["search"]

    out = search("framework")            # 3 ranked < the default k=8

    note = next(n for n in out["notices"] if n["code"] == "thin_coverage")
    assert note["ranked"] == 3 and note["asked"] == 8
    assert "watch(action='preview')" in note["message"] and "web-search" in note["message"]
    assert "`look`" in note["message"]                # a named thing, searched outside
    assert "once per conversation" in note["message"]

    again = search("framework")
    assert "thin_coverage" in [n["code"] for n in again["notices"]]


def test_a_filtered_thin_search_is_thin_on_purpose_and_gets_no_offer(kb_home):
    """A filter narrows deliberately — few results under `who=` or a date bound is the caller's
    own scoping, not a coverage gap in the store."""
    conn = schema.connect(); _corpus(conn); conn.close()
    search = _tools()["search"]

    out = search("framework", source_type="paper")

    assert "thin_coverage" not in [n["code"] for n in out["notices"]]


def test_a_search_that_fills_its_ask_gets_no_offer(kb_home):
    conn = schema.connect(); _corpus(conn); conn.close()
    search = _tools()["search"]

    out = search("framework", k=2)       # 3 ranked >= the 2 asked

    assert "thin_coverage" not in [n["code"] for n in out["notices"]]


def test_zero_matches_is_the_strongest_thin_signal_and_still_fires(kb_home):
    conn = schema.connect(); _corpus(conn); conn.close()
    search = _tools()["search"]

    out = search("sim-to-real transfer in robotics")

    note = next(n for n in out["notices"] if n["code"] == "thin_coverage")
    assert note["ranked"] == 0


# ── thin ≠ unfinished ──────────────────────────────────────────────
#
# An import that has not run produces the same arithmetic as a genuinely thin store, and the
# offer above reads that arithmetic as a diagnosis: "the user has just named a subject their store
# is thin on". On a store whose bookmark import is still queued that is confidently wrong, and
# expensively so — it asks the user to go web-search for material they already handed OPYT.

def _begin_import(finish=False):
    """The Bookmarks walk begun in this home's store (an import promised), or walked to the end."""
    from datetime import datetime, timezone

    from pipeline.curation import bookmark_listing, listings
    from pipeline.platforms.collection_page import Page

    conn = schema.connect()
    try:
        listings.init_schema(conn)
        now = datetime.now(timezone.utc)
        walk = listings.start(conn, bookmark_listing.COLLECTOR, "me", listings.Branch("Bookmarks"),
                              now)
        if finish:
            task = listings.pending(conn, walk)[0]
            listings.accept_page(conn, walk, task, Page((), None, (), ended=True))
            listings.complete(conn, walk, now)
    finally:
        conn.close()


def test_a_thin_search_blames_the_unfinished_import_not_the_user(kb_home):
    conn = schema.connect(); _corpus(conn); conn.close()
    _begin_import()
    search = _tools()["search"]

    out = search("framework")
    codes = [n["code"] for n in out["notices"]]

    assert "import_incomplete" in codes and "thin_coverage" not in codes
    note = next(n for n in out["notices"] if n["code"] == "import_incomplete")
    assert note["waiting"] == ["bookmarks_import"] and note["ranked"] == 3
    # The host must be told NOT to make the offer, not merely left without the wording for it.
    assert "not" in note["message"].lower() and "offer nothing else" in note["message"]


def test_an_import_that_has_landed_is_not_an_outstanding_promise(kb_home):
    """An import walked to the end with nothing owed is a real observation: the store IS thin,
    and suppressing the offer there would lose the highest-intent moment there is."""
    conn = schema.connect(); _corpus(conn); conn.close()
    _begin_import(finish=True)

    out = _tools()["search"]("framework")
    codes = [n["code"] for n in out["notices"]]

    assert "thin_coverage" in codes and "import_incomplete" not in codes


def test_nothing_outstanding_leaves_the_offer_alone(kb_home):
    """The mirror of the import rule: with nothing owed, a thin result is a real observation
    about the store, and suppressing the offer forever would lose the highest-intent moment there
    is to a flag nobody clears."""
    conn = schema.connect(); _corpus(conn); conn.close()

    codes = [n["code"] for n in _tools()["search"]("framework")["notices"]]
    assert "thin_coverage" in codes and "import_incomplete" not in codes


def test_an_oracle_pull_still_owed_is_named_instead_of_thin_coverage(kb_home):
    """Right after onboarding the Oracle pulls are still owed — a pick's
    newest posts, its archives — and a thin search read that as "you are thin on this, go enrich
    it" while the material was simply not in yet. What is owed is the milestones' whole list,
    the saved-post imports included."""
    from pipeline.oracles import registry
    from pipeline.oracles.sources import scraper_tables
    from pipeline.oracles import asks
    from pipeline.oracles import windows
    from pipeline.timeparse import utc_now

    conn = schema.connect()
    _corpus(conn)
    registry.init_state_schema(conn)
    scraper_tables.init_schema(conn)
    roster.upsert_oracle(conn, "x:user:andrej", name="Andrej")
    registry.upsert_source(conn, registry.SourceRow("x:user:andrej", "x", "andrej"))
    asks.record_ask(conn, "x:user:andrej", web_from=windows.ALL_TIME)
    conn.execute("UPDATE oracle_asks SET discovered_at=?", (utc_now().isoformat(),))
    conn.commit()
    conn.close()

    out = _tools()["search"]("framework")
    codes = [n["code"] for n in out["notices"]]
    assert "import_incomplete" in codes and "thin_coverage" not in codes
    note = next(n for n in out["notices"] if n["code"] == "import_incomplete")
    assert "newest_posts" in note["waiting"]


def test_another_homes_import_is_not_this_homes_promise(tmp_path, monkeypatch):
    """On the hosted shared queue, listing every home's rows would let one user's queued
    import make another user's thin search say `import_incomplete`. An import is read from the
    home's own store, so another home's cannot appear."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "B"))
    _begin_import()
    assert imports.imports_outstanding() == ["bookmarks_import"]

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "A"))
    assert imports.imports_outstanding() == []


def test_an_unreadable_import_state_leaves_the_search_alone(kb_home, monkeypatch):
    """Fail-safe: a missing or broken optional input degrades to an empty result, and
    a notice is never worth failing a query over. Reporting "nothing outstanding" is also the
    right direction to be wrong in — search keeps its normal behavior."""
    conn = schema.connect(); _corpus(conn); conn.close()
    _begin_import()
    monkeypatch.setattr(imports, "_states",
                        lambda: (_ for _ in ()).throw(RuntimeError("store gone")))

    assert imports.imports_outstanding() == []
    out = _tools()["search"]("framework")
    assert "thin_coverage" in [n["code"] for n in out["notices"]]


# ── `cite` — the pre-rendered link ──────────────────────────────────────────────
# `source_url` is on every card and a
# docstring told the host to show it, and two hosted phone searches still came back linkless
# because a bare URL doubles the width of a short bullet. `cite` makes the link cost nothing —
# the title the host was already writing becomes the link. See `pipeline.retrieval.cards.cite`.
def test_every_hit_carries_a_pasteable_markdown_link(kb_home):
    """The whole point of the field: a host can paste it and be done."""
    conn = schema.connect(); _corpus(conn); conn.close()
    hits = kb_entry.run_kb_search("framework", k=8)["hits"]
    assert hits
    for h in hits:
        assert h["cite"].startswith("[") and f"]({h['source_url']})" in h["cite"]


def test_cite_never_replaces_source_url(kb_home):
    """Presentation is added, never substituted. Anything that programs against a hit reads the
    bare pointer, and collapsing the two would make a formatting choice load-bearing for every
    non-host consumer of this API."""
    conn = schema.connect(); _corpus(conn); conn.close()
    h = kb_entry.run_kb_search("framework", k=8)["hits"][0]
    assert h["source_url"] and "[" not in h["source_url"]


def test_an_atom_with_no_url_cites_nothing_rather_than_nowhere(kb_home):
    """Fail-safe, and the failure it prevents is specific: `[label]()` renders as a link, reads
    as a citation, and goes nowhere. `None` is the honest answer and the docstring tells the host
    to say so plainly."""
    assert cards.cite(None, "NASA", "x", "2026-06-30") is None
    assert cards.cite("", "NASA", "x", "2026-06-30") is None


def test_a_display_name_with_a_bracket_cannot_break_out_of_the_label(kb_home):
    """X allows `]` in a display name. Unescaped it closes the markdown label early and spills
    the rest of the name into the output as loose text beside a truncated link."""
    out = cards.cite("https://x.com/a/status/1", "Bob [not] Real", "x", "2026-06-30")
    assert out == "[Bob not Real · 2026-06-30](https://x.com/a/status/1)"


def test_a_url_with_parens_is_wrapped_so_the_target_survives(kb_home):
    """`(disambiguation)`-style URLs are the standard case. A bare paren ends the markdown target
    early, producing a link that silently points somewhere SHORTER than intended — which is worse
    than no link, because it looks like it worked."""
    out = cards.cite("https://en.wikipedia.org/wiki/Mercury_(planet)", None, "blog", None)
    assert out == "[blog](<https://en.wikipedia.org/wiki/Mercury_(planet)>)"


def test_cite_falls_back_through_name_then_type_then_the_word_source(kb_home):
    """A label is never empty — an atom with no author name and no source type still gets a
    clickable word rather than `[](url)`."""
    assert cards.cite("https://e/1", None, None, None) == "[source](https://e/1)"
    assert cards.cite("https://e/1", None, "paper", None) == "[paper](https://e/1)"


def test_open_cites_the_same_way_a_hit_does(kb_home):
    """The surface where a claim is actually asserted must spell its citation the same way the
    surface that routed you there did, or the host learns two habits and keeps the cheaper one."""
    conn = schema.connect(); _corpus(conn); conn.close()
    got = kb_entry.kb_open(C)
    assert got["cite"] == f"[x · 2024-05-01](https://example/{C})"
