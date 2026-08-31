"""`run_kb_search`'s response envelope — `{hits, notices, insights, trace, frontier_atoms}`.

Search used to return a bare `list[dict]`. A list has no header, so there was nowhere to
report a fact ABOUT the result — only the result itself — and facts that had to go somewhere
ended up on the wrong tool (`resolved_who` on `aggregate`) or nowhere at all (which arms ran,
what units `score` is in, what a filter cost you). These pin the keys, and specifically
pin the silent behaviours that had no reporting channel before 2026-08-04.

`frontier_atoms` joined 2026-08-27: a default call ranks its full k over HUMAN_ATTESTED atoms
and answers the keyword-crawl population BESIDE that list rather than inside it.

The rule under test throughout: `notices` are SENTENCES (repeat them to the user),
`insights`/`trace` are VALUES (reason over them, never recite).
"""
from __future__ import annotations

import pytest

from mcp_server import atoms_tools
from opyt_core import kb as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot

A = "github:root/agentkit"
B = "github:stranger/agents"
C = "x:1"
P = "paper:2401.00001"


def _add(conn, emb, atom_id, source_type, what_kind, who_id, topics, snapshot,
         *, entry_mode="oracle-footprint", when_ts="2024-05-01", body_state=None):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, snapshot)
    payload = {"source_tags": topics}
    if body_state is not None:
        payload["body_state"] = body_state
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts=when_ts, when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card", payload=payload, entry_mode=entry_mode,
    ), snapshot_text=snapshot)


def _corpus(conn, emb):
    _add(conn, emb, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents",
         entry_mode="user-saved", when_ts="2024-05-01", body_state="complete")
    _add(conn, emb, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", entry_mode="oracle-footprint", when_ts="2024-07-09",
         body_state="partial")
    _add(conn, emb, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems", entry_mode="user-saved")
    _add(conn, emb, P, "paper", "artifact", "scholar:1", ["ai-agents"],
         "a paper about an agent framework and its library", entry_mode="oracle-footprint")


@pytest.fixture()
def local_embedder(monkeypatch, fake_embedder):
    """Make `run_kb_search` use the offline bag-of-words embedder.

    `mode="semantic"`/`"hybrid"` build the REAL hosted embedder, which is a paid call AND a
    different subspace from the one the fixtures ingested with — so the subspace guard rejects
    it before any assertion runs. The tests that need this are about `trace`, not about
    embedding, so the arm just has to RUN, not to be any good."""
    monkeypatch.setattr(kb_entry, "get_kb_embedder", lambda: fake_embedder)
    return fake_embedder


@pytest.fixture(autouse=True)
def _clean_session():
    """Session counters are MODULE globals (stdio = one process per session), so they leak
    from test to test unless cleared. Both sides: a test must not inherit a count, and must
    not leave one behind."""
    atoms_tools._reset_session()
    yield
    atoms_tools._reset_session()


# ── the shape ────────────────────────────────────────────────────────────────────

def test_an_ordinary_query_returns_all_five_keys(kb_home, fake_embedder):
    """`frontier_atoms` joined the envelope 2026-08-27 and is present on every DEFAULT call —
    including one that found no frontier atoms, so "none matched" and "this build has no such
    list" cannot look the same. Scoping the call to a population removes it: the whole answer
    is then that population and a second list would be the same list."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert set(out) == {"hits", "notices", "insights", "trace", "frontier_atoms"}
    assert out["frontier_atoms"]["hits"] == []          # the corpus is all human-attested

    scoped = kb_entry.run_kb_search("framework", entry_mode="frontier", mode="bm25", k=8)
    assert set(scoped) == {"hits", "notices", "insights", "trace"}


def test_hits_are_exactly_what_the_bare_list_used_to_hold(kb_home, fake_embedder):
    """The envelope wraps; it must not move ranking or hit content. Same ids, same order, same
    card keys as before — everything new lives OUTSIDE `hits`."""
    conn = schema.connect(); _corpus(conn, fake_embedder)
    from pipeline.kb.retrieve import search_atoms
    raw = search_atoms(conn, "framework", fake_embedder, k=8, mode="bm25")
    conn.close()

    hits = kb_entry.run_kb_search("framework", mode="bm25", k=8)["hits"]
    assert [h["atom_id"] for h in hits] == [h.atom_id for h in raw.hits]
    assert {"citation_id", "atom_id", "snippet", "source_url", "score", "body_state",
            "payload"} <= set(hits[0])


def test_a_healthy_query_reports_no_notices(kb_home, fake_embedder):
    """THE contract for `notices`: emptiness is meaningful. If a clean query carried chatter,
    a host would learn to skip the field and would then skip the real ones too."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    assert kb_entry.run_kb_search("framework", mode="bm25", k=8)["notices"] == []


# ── the step-0 bug, asserted through the envelope ────────────────────────────────

def test_a_tag_that_normalizes_away_matches_nothing_and_says_so(kb_home, fake_embedder):
    """`slugify("+++")` is empty, so this used to leave the pre-filter with no clauses at all
    and return THE WHOLE STORE under a filter the caller believed was applied. Two assertions,
    because they are two separate failures: returning everything is the bug, and staying quiet
    about the dropped tag is what made it invisible."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["+++"], mode="bm25", k=8)

    assert out["hits"] == []                                # NOT the whole store
    codes = [n["code"] for n in out["notices"]]
    assert "tags_normalized" in codes
    assert "+++" in next(n for n in out["notices"] if n["code"] == "tags_normalized")["message"]


def test_a_mangled_tag_is_reported_even_though_it_still_matches(kb_home, fake_embedder):
    """The half the pre-filter fix CANNOT catch. `"C++"` slugifies to `"c"` — a valid, entirely
    different filter — so the query runs and succeeds while answering a question nobody asked.
    A silent substitution has no symptom; only the notice makes it visible."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["C++"], mode="bm25", k=8)

    note = next(n for n in out["notices"] if n["code"] == "tags_normalized")
    assert note["changed"] == [["C++", "c"]]
    assert out["trace"]["filters"]["tags"] == ["c"]          # filters as APPLIED, not as asked


def test_a_cost_sentence_names_a_filter_value_only_when_there_is_one(kb_home, fake_embedder):
    """A notice is read out to the user verbatim, so it has to survive the degenerate case. A
    filter that normalized away has an applied value of `[]`, and "Dropping tags=[] would
    return 4" is noise — the sentence explaining WHY it is empty is a separate notice."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["+++"], mode="bm25", k=8)

    msg = next(n for n in out["notices"] if n["code"] == "filters_matched_nothing")["message"]
    assert "tags=[]" not in msg
    assert "Dropping tags would return 4." in msg


# ── empty results have four different causes ─────────────────────────────────────

def test_an_empty_store_says_so_rather_than_returning_a_bare_empty(kb_home, fake_embedder):
    schema.connect().close()                                 # schema only, zero atoms
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert out["hits"] == []
    assert [n["code"] for n in out["notices"]] == ["store_empty"]
    # A dead end is worse than no notice: whoever reads this has an empty store and no idea what
    # to do about it, and the actual blocker (two unacquired API keys) is nowhere in the sentence.
    note = out["notices"][0]
    assert note["next_tool"] == "onboard"
    assert "onboard" in note["message"]


def test_filters_that_match_nothing_report_what_the_filter_cost(kb_home, fake_embedder):
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["ai-agents"], source_type="substack",
                                 mode="bm25", k=8)

    note = next(n for n in out["notices"] if n["code"] == "filters_matched_nothing")
    assert out["insights"]["filter_cost"]["source_type"] == 3     # A, B, P are ai-agents
    assert "source_type" in note["message"]
    assert out["trace"]["ran"] == "none"                          # no arm ever ran


def test_an_unresolvable_handle_is_distinguishable_from_a_quiet_person(kb_home, fake_embedder):
    """The two `[]`s that used to look identical, now separated in ONE call."""
    conn = schema.connect(); _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")
    conn.close()

    untracked = kb_entry.run_kb_search("framework", who="@nobody", mode="bm25", k=8)
    assert [n["code"] for n in untracked["notices"]] == ["who_unresolved"]
    assert "@nobody" in untracked["notices"][0]["message"]
    # This is the local-only tool's only channel back to a host that has web search and this
    # one doesn't — a miss must point the host at it, not just report the miss.
    assert "web-search" in untracked["notices"][0]["message"]
    assert untracked["insights"]["resolved_who"] == []

    quiet = kb_entry.run_kb_search("dashboard", who="root", mode="bm25", k=8)
    assert quiet["hits"] == []                                   # they wrote nothing on it...
    assert quiet["insights"]["resolved_who"][0]["name"] == "Root McRepo"   # ...but we HAVE them
    assert "who_unresolved" not in [n["code"] for n in quiet["notices"]]


def test_a_handle_matching_two_people_warns_that_results_mix_them(kb_home, fake_embedder):
    """`resolve_who` returns one entry per CLUSTER. Two clusters means the hits are two
    different people's work presented as one person's — worth a sentence, not a value."""
    conn = schema.connect(); _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Root One")
    schema.upsert_entity(conn, "substack:root", name="Root Two")
    conn.close()

    out = kb_entry.run_kb_search("framework", who="root", mode="bm25", k=8)
    note = next(n for n in out["notices"] if n["code"] == "who_multiple")
    assert len(note["canonical_ids"]) == 2
    assert "Root One" in note["message"] and "Root Two" in note["message"]


# ── filter_cost is self-pruning, which is what replaces a threshold ───────────────

def test_filter_cost_is_empty_when_every_filter_is_load_bearing(kb_home, fake_embedder):
    """No "is this result thin enough to explain" threshold exists anywhere, because a
    well-targeted query prices out at `{}` BY CONSTRUCTION. `what_kind="artifact"` excludes
    only C, which the tag filter already excluded — so dropping it would gain nothing."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", tags=["ai-agents"], what_kind="artifact",
                                 mode="bm25", k=8)
    assert out["insights"]["filter_cost"] == {}


def test_filter_cost_is_reported_even_when_the_query_looks_healthy(kb_home, fake_embedder):
    """The dangerous case, and the reason this is computed on EVERY search rather than only on
    thin ones: a filter narrows the result, the hits still look clean, and nothing anywhere
    says what was excluded. A silent narrowing has no symptom to trigger on."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", source_type="github", mode="bm25", k=8)

    assert len(out["hits"]) == 2                       # a perfectly healthy-looking result...
    assert out["insights"]["filter_cost"]["source_type"] == 2   # ...that excluded C and P


# ── trace: the three silent engine behaviours ────────────────────────────────────

def test_hybrid_reports_when_it_actually_ran_one_arm(kb_home, fake_embedder, local_embedder):
    """The docstring claimed "both fire in hybrid". `bm25_weight` is 0.0 for any conceptual
    query over three words, and the keyword arm is then skipped entirely — which is the MOST
    COMMON query shape an MCP host sends. Pinned so the claim cannot quietly come back."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("what does david think about agent memory",
                                 mode="hybrid", k=8)
    assert out["trace"]["ran"] == "semantic"           # asked for hybrid, ran ONE arm
    assert "bm25_weight 0.0" in out["trace"]["why"]
    assert out["trace"]["fts_query"] is None           # the keyword arm never ran


def test_score_scale_names_three_different_units_under_one_key(kb_home, fake_embedder,
                                                               local_embedder):
    """`score` means three things. Comparing 0.03 from one call against 0.72 from another is
    comparing different units, and nothing said so until this field existed."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    q = "agent framework library"                      # literal tokens → hybrid runs BOTH arms
    scales = {m: kb_entry.run_kb_search(q, mode=m, k=8)["trace"]["score_scale"]
              for m in ("bm25", "semantic", "hybrid")}
    assert scales == {"bm25": "reciprocal_rank", "semantic": "cosine", "hybrid": "rrf"}


def test_the_rewritten_keyword_query_is_visible(kb_home, fake_embedder):
    """A BM25 hit needs ONE token, not the phrase — `_fts_query` ORs them. That rewrite
    explains hits that otherwise look irrelevant, and nobody could see it."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework", mode="bm25", k=8)
    assert out["trace"]["fts_query"] == '"agent" OR "framework"'


def test_truncation_reports_the_score_at_the_boundary(kb_home, fake_embedder):
    """THE instrument for "did top-k cut off good answers". `candidates` is a FILTER fact — how
    many cleared the pre-filter, most possibly irrelevant. The two scores at the cut are the
    RELEVANCE fact: near-identical means the cut landed mid-cluster, a large drop means k was
    already the right number. Reported, never judged — no threshold here."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=2)

    assert len(out["hits"]) == 2
    assert out["trace"]["ranked"] == 3                 # A, B, P all matched
    cut = out["trace"]["cutoff"]
    assert cut["last_returned"] == out["hits"][-1]["score"]
    assert cut["next_best"] < cut["last_returned"]     # rank 3's score, the one you didn't get
    assert [n["code"] for n in out["notices"]].count("results_truncated") == 1


def test_cutoff_is_absent_when_nothing_was_cut(kb_home, fake_embedder):
    """ABSENT, not zeroed. There was no boundary to describe, and a 0.0 would read as one."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=8)
    assert "cutoff" not in out["trace"]
    assert [n["code"] for n in out["notices"]] == []


def test_candidates_separates_unfiltered_from_filtered_to_nothing(kb_home, fake_embedder):
    """`None` (no filter ran, whole store searched) and `0` (a filter ran, nothing survived)
    are opposite facts, so they must not share a value.

    `None` is no longer reachable THROUGH `run_kb_search`: since the sectioned response
    (2026-08-27) a default call always scopes to HUMAN_ATTESTED, so a count is always a real
    count. The distinction still has to hold one layer down, where an unscoped search lives."""
    conn = schema.connect(); _corpus(conn, fake_embedder)
    from pipeline.kb.retrieve import search_atoms
    assert search_atoms(conn, "framework", fake_embedder, mode="bm25").candidates is None
    conn.close()
    assert kb_entry.run_kb_search("framework", mode="bm25")["trace"]["candidates"] == 4
    assert kb_entry.run_kb_search("framework", source_type="youtube",
                                  mode="bm25")["trace"]["candidates"] == 0


# ── insights: distributions over the hits, reported not judged ───────────────────

def test_insights_summarize_exactly_the_hits_they_describe(kb_home, fake_embedder):
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=8)
    ins, hits = out["insights"], out["hits"]

    assert sum(ins["sources"].values()) == len(hits)
    assert ins["sources"] == {"github": 2, "paper": 1}
    assert ins["body_state"]["complete"] == 1 and ins["body_state"]["partial"] == 1
    assert ins["date_span"] == ["2024-05-01", "2024-07-09"]


def test_entry_mode_reaches_insights_from_a_column_no_query_ever_read(kb_home, fake_embedder):
    """`atoms.entry_mode` has been written since the beginning and read by ZERO query paths.
    'You saved this yourself' vs 'we swept it because we track the author' is real provenance,
    and it was sitting in the row the whole time.

    The tally is a PASS-THROUGH of whatever modes appear, not a two-way split — the key name
    predates `oracle-footprint` and `author_referenced` and is now misleading. Renaming it is an
    MCP-payload change, filed separately."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library rollup", mode="bm25", k=8)
    assert out["insights"]["saved_vs_crawled"] == {"oracle-footprint": 2, "user-saved": 2}


def test_a_lopsided_result_is_a_distribution_never_a_warning(kb_home, fake_embedder):
    """Fork 4, asserted. Every hit here is by one author. `insights` reports the concentration
    and the HOST decides whether that is alarming; `notices` must NOT editorialize about
    evidence quality, because that is a judgement the tool has no standing to make."""
    conn = schema.connect(); _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")
    conn.close()

    out = kb_entry.run_kb_search("autonomous", mode="bm25", k=8)
    assert out["insights"]["authors"] == {"Root McRepo": 1}
    assert out["notices"] == []                        # a distribution, not a warning


def test_corpus_newest_is_the_whole_store_not_the_hits(kb_home, fake_embedder):
    """A fact no hit can carry: "my newest atom is months old" is what explains a result that
    looks stale, and it is invisible from inside a list of hits."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert out["insights"]["corpus_newest"] is not None


# ── session state (module-level, MCP layer only) ─────────────────────────────────

class _Mcp:
    """A stand-in for FastMCP that just keeps the decorated functions."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def _tools():
    m = _Mcp()
    atoms_tools.register_atoms_tools(m)
    return m.tools


def test_three_searches_without_opening_anything_says_so(kb_home, fake_embedder):
    """A PROCESS fact, not a judgement about evidence — searching alone never grounds a claim.
    It arrives at the moment the pattern occurs, where a static docstring line would have
    diluted across a long context."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    search = _tools()["search"]

    assert [n["code"] for n in search("framework", mode="bm25")["notices"]] == []
    assert [n["code"] for n in search("agent", mode="bm25")["notices"]] == []
    third = search("library", mode="bm25")
    assert "nothing_grounded" in [n["code"] for n in third["notices"]]
    assert third["trace"]["session"]["search_n"] == 3


def test_one_open_clears_the_nudge(kb_home, fake_embedder):
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    tools = _tools()
    for _ in range(2):
        tools["search"]("framework", mode="bm25")
    tools["open"](A)

    out = tools["search"]("library", mode="bm25")
    assert "nothing_grounded" not in [n["code"] for n in out["notices"]]
    assert out["trace"]["session"]["opened_so_far"] == 1


def test_overlap_is_reported_only_when_it_is_a_signal(kb_home, fake_embedder):
    """Two unrelated searches sharing one atom is noise. A repeated SET is the signal — the
    same evidence circled without opening it."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    search = _tools()["search"]

    first = search("agent framework library", mode="bm25")
    assert first["trace"]["session"]["overlap_with_recent"] is None        # nothing before it
    again = search("agent framework library", mode="bm25")
    assert again["trace"]["session"]["overlap_with_recent"] >= 2


def test_overlap_looks_back_further_than_the_single_previous_search(kb_home, fake_embedder):
    """Named `overlap_with_recent`, not `_with_previous`, because it means it. Circling the
    same evidence is rarely consecutive — A, B, A is the ordinary shape — and a strictly
    previous-only comparison would miss exactly the pattern worth catching."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    search = _tools()["search"]

    search("agent framework library", mode="bm25")     # A
    search("rollup proof", mode="bm25")                 # B — an unrelated detour
    back = search("agent framework library", mode="bm25")   # A again
    assert back["trace"]["session"]["overlap_with_recent"] >= 2


def test_a_failed_open_does_not_count_as_grounding(kb_home, fake_embedder):
    """`open` on an unknown id returns an error dict and grounds nothing. Counting it would let
    a typo silence the one nudge that says "you have not read a source yet"."""
    conn = schema.connect(); _corpus(conn, fake_embedder); conn.close()
    tools = _tools()
    tools["open"]("github:nope/nope")

    for _ in range(3):
        out = tools["search"]("framework", mode="bm25")
    assert "nothing_grounded" in [n["code"] for n in out["notices"]]


# ── the freshness notice rides on `search`, gated on needs_attention ─────────
#
# Decision 4 said freshness rides on `search` AND `oracle(action="screen")`. Only `screen`
# shipped. Users go to `search`; they rarely go to `screen`, and a frozen roster has to be
# visible where people actually are.

def test_stale_oracles_are_reported_on_search(kb_home, fake_embedder, monkeypatch):
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "status_summary",
                        lambda conn: {"needs_attention": True, "stale_fraction": 0.9})
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    note = next(n for n in out["notices"] if n["code"] == "oracles_stale")
    assert "refreshed" in note["message"]


def test_fresh_oracles_add_no_notice(kb_home, fake_embedder, monkeypatch):
    """⚠️ GATED, NOT UNCONDITIONAL — unlike `screen`, which is deliberate and occasional.
    `search` is high-frequency, and a block printed on every call trains the reader to skip
    the one call where it matters."""
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "status_summary",
                        lambda conn: {"needs_attention": False})
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert "oracles_stale" not in [n["code"] for n in out["notices"]]


# ── a budget-paused rail rides on `search` too, gated on there being one ────
#
# Same argument as the freshness notice one section up, one failure mode over. A rail that hits
# its daily ceiling writes `budget_paused` to ~/.opyt/<rail>.log, which nothing reads — so the
# rail goes quiet and looks broken. That is the frozen-Oracle shape: refusing correctly, in
# private. `search` is where the user actually is when their corpus stops growing.

def test_a_budget_paused_rail_is_reported_on_search(kb_home, fake_embedder, monkeypatch):
    from pipeline.kb import rail_budgets
    monkeypatch.setattr(rail_budgets, "paused_today",
                        lambda: [{"rail": "bookmark_catchup", "label": "bookmark catch-up",
                                  "spent_usd": 1.0, "ceiling_usd": 1.0}])
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    note = next(n for n in out["notices"] if n["code"] == "rails_budget_paused")
    assert "bookmark catch-up" in note["message"]
    assert note["rails"][0]["rail"] == "bookmark_catchup"


def test_no_paused_rail_adds_no_notice(kb_home, fake_embedder, monkeypatch):
    """⚠️ GATED ON A REAL PAUSE, not printed every call. `search` is high-frequency, and a block
    on every call trains the reader to skip the one call where it matters."""
    from pipeline.kb import rail_budgets
    monkeypatch.setattr(rail_budgets, "paused_today", lambda: [])
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert "rails_budget_paused" not in [n["code"] for n in out["notices"]]


def test_a_spend_meter_hiccup_never_breaks_a_search(kb_home, fake_embedder, monkeypatch):
    from pipeline.kb import rail_budgets
    monkeypatch.setattr(rail_budgets, "paused_today",
                        lambda: (_ for _ in ()).throw(RuntimeError("stats file is a directory")))
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    assert kb_entry.run_kb_search("framework", mode="bm25", k=8)["hits"]


def test_a_freshness_hiccup_never_breaks_a_search(kb_home, fake_embedder, monkeypatch):
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "status_summary",
                        lambda conn: (_ for _ in ()).throw(RuntimeError("db locked")))
    conn = schema.connect()
    _add(conn, fake_embedder, A, "github", "repo", "who:1", ["agents"], "an agent framework")
    conn.close()
    assert kb_entry.run_kb_search("framework", mode="bm25", k=8)["hits"]


# ── the sectioned response (RULED 2026-08-27) ────────────────────────────────────
# A frontier atom's max-pooled cosine is not comparable to a human-attested one's — papers are
# 24x denser than X posts, and nobody DECIDED that a result set should be mostly papers. So the
# two populations are ranked apart and labeled, and no score crosses between them.

def _with_frontier(conn, emb):
    """The `_corpus` atoms plus three frontier ones whose scores fall off a cliff after the head,
    so the block's own-top floor has a tail to remove."""
    _corpus(conn, emb)
    _add(conn, emb, "paper:crawl/strong", "paper", "artifact", "openalex:1", ["ai-agents"],
         "an agent framework library", entry_mode="frontier")
    _add(conn, emb, "paper:crawl/mid", "paper", "artifact", "openalex:2", ["ai-agents"],
         "an agent framework library for tools", entry_mode="frontier")
    _add(conn, emb, "github:crawl/weak", "github", "artifact", "github:crawler", ["crypto"],
         "a rollup proof dashboard", entry_mode="frontier")


def test_a_default_call_ranks_humans_alone_and_labels_the_crawl_beside_them(kb_home,
                                                                            fake_embedder):
    """THE decision, pinned. `hits` holds no frontier atom however well it scored, and the crawl
    is not dropped either — it is returned under its own key, where a host can offer it as a
    different KIND of evidence rather than as more of the same."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=8)

    assert out["hits"], "the human-attested list must still be the answer"
    assert all(h["entry_mode"] != "frontier" for h in out["hits"])
    assert {h["atom_id"] for h in out["frontier_atoms"]["hits"]} <= {
        "paper:crawl/strong", "paper:crawl/mid", "github:crawl/weak"}
    assert out["frontier_atoms"]["candidates"] == 3          # cleared the frontier scope


def test_an_explicit_scope_returns_one_list_and_no_second_one(kb_home, fake_embedder):
    """"More frontier" needed no new tool and no new surface — it is the same call, scoped.
    Pinned because the alternative (a separate frontier search tool) was rejected on the
    measured cost of host tool-selection past ~30 tools."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", entry_mode="frontier",
                                 mode="bm25", k=20)

    assert out["hits"] and all(h["entry_mode"] == "frontier" for h in out["hits"])
    assert "frontier_atoms" not in out
    assert out["trace"]["filters"]["entry_mode"] == "frontier"   # applied, and said so


def test_the_block_floor_drops_the_weak_tail_and_reports_the_drop(kb_home, fake_embedder,
                                                                  local_embedder):
    """The floor is a fraction of the BLOCK's own top, never of the human list's — comparing
    those two tops is exactly the cross-provenance comparison this design refuses. And the
    drop is COUNTED: a block that cut its tail must not read like one that had none.

    Cosines under the FakeEmbedder: strong 1.000, mid 0.866, weak 0.000. At 0.80 the first two
    clear and the third does not."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    block = kb_entry.run_kb_search("agent framework library", mode="semantic",
                                   k=8)["frontier_atoms"]

    assert {h["atom_id"] for h in block["hits"]} == {"paper:crawl/strong", "paper:crawl/mid"}
    assert block["floor"] == {"frac": kb_entry.FRONTIER_BLOCK_FLOOR_FRAC, "applied": True,
                              "score_scale": "cosine", "top": 1.0, "dropped": 1}


def test_the_floor_does_not_run_on_a_scale_that_measures_position(kb_home, fake_embedder):
    """`reciprocal_rank` is 1/(1+rank) — the BM25 relevance value orders the rows and is then
    discarded, so `score` is the loop counter. EVERY floor from 0.51 to 1.00 keeps exactly rank
    0, whatever it matched, which is `hits[:1]` wearing a quality floor's name. `rrf` fails the
    other way: it is a SUM across arms, so a single-arm hit scores 0.25-0.50 of a dual-arm one
    and a 0.80 floor would cut atoms for which retrieval method found them.

    Both are skipped, and `applied` says so — "ran and cut nothing" and "could not run" both
    leave `dropped: 0`, and those are opposite facts."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    block = kb_entry.run_kb_search("agent framework library", mode="bm25",
                                   k=8)["frontier_atoms"]

    assert block["floor"]["applied"] is False
    assert block["floor"]["score_scale"] == "reciprocal_rank"
    assert block["floor"]["dropped"] == 0
    assert len(block["hits"]) > 1                       # nothing was cut on a positional score


def test_a_broken_frontier_pass_costs_the_answer_nothing(kb_home, fake_embedder, monkeypatch):
    """FAIL-SAFE, the project invariant. The block is an ADDITION to the answer; a query that
    would have worked before it existed must still work. Degrade to an empty block plus a
    notice — never a raised exception, and never a silently missing key."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()

    def _boom(*a, **kw):
        raise RuntimeError("frontier pass exploded")
    monkeypatch.setattr(kb_entry, "_frontier_block", _boom)
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=8)

    assert out["hits"]                                    # the real answer survived
    assert out["frontier_atoms"]["hits"] == []
    note = next(n for n in out["notices"] if n["code"] == "frontier_block_failed")
    assert "RuntimeError" in note["error"]


def test_a_card_says_which_list_it_came_from(kb_home, fake_embedder):
    """`entry_mode` is user-facing POLICY now, not just a filter column. Cards get copied out of
    their list — into a citation, a follow-up call — and one read alone must still say whether a
    person vouched for it or a keyword query found it."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("agent framework library", mode="bm25", k=8)

    assert {h["entry_mode"] for h in out["hits"]} <= set(schema.HUMAN_ATTESTED)
    assert all(h["entry_mode"] == "frontier" for h in out["frontier_atoms"]["hits"])


class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_the_search_tool_exposes_the_scope_and_keeps_frontiers_queue_notice_apart(
        kb_home, fake_embedder, monkeypatch):
    """Two different things are called "frontier" in one response and they must not collide:
    `frontier_atoms` is atoms already in the KB that matched this query; `frontier` is the
    QUEUE's once-a-session push about staged candidates that are NOT in the KB yet."""
    conn = schema.connect(); _with_frontier(conn, fake_embedder); conn.close()
    monkeypatch.setattr("mcp_server.frontier_tools.notice", lambda: {"unshown": 7})

    mcp = _FakeMCP()
    atoms_tools.register_atoms_tools(mcp)
    out = mcp.tools["search"]("agent framework library", mode="bm25", k=8)

    assert out["frontier"] == {"unshown": 7}               # the queue's push, untouched
    assert "hits" in out["frontier_atoms"]                 # this query's crawl results
    scoped = mcp.tools["search"]("agent framework library", entry_mode="frontier", mode="bm25")
    assert "frontier_atoms" not in scoped
