"""retrieve.search_atoms — the search mechanics, proven offline: the tag pre-filter (over
`payload.source_tags`) and the one BM25 arm over whole item bodies.

Ranking is pure relevance: no trust re-rank applies, and
`test_ranking_is_pure_relevance_not_trust` below asserts that.
"""
from __future__ import annotations

from pipeline.retrieval import reads as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.retrieval.engine import search_atoms
from pipeline.retrieval.filters import candidate_atom_ids, filter_costs

# ── corpus ──────────────────────────────────────────────────────────────────────
A = "github:root/agentkit"   # ai-agents, rich text
B = "github:stranger/agents"  # ai-agents, terse text — the tighter match
C = "x:1"                      # crypto opinion
D = "github:root/dash"        # web


def _add(conn, atom_id, source_type, what_kind, who_id, topics, snapshot,
         source_tags=None, entry_mode="user-saved"):
    raw_hash = snapshot_hash(snapshot)
    tags = list(topics) + list(source_tags or [])
    atom = dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts="2024-05-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_hash=raw_hash,
        description=f"{atom_id} card",
        # `source_tags` absent unless there is one — the realistic shape, since most sources
        # declare no labels. The absent key is what the filter must tolerate (json_each over
        # a missing key yields zero rows rather than erroring).
        payload={"source_tags": tags} if tags else {},
        entry_mode=entry_mode,
    )
    store_atom(conn, atom, snapshot)


def _corpus(conn):
    _add(conn, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents")
    _add(conn, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library")
    _add(conn, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems")
    _add(conn, D, "github", "artifact", "github:root", ["web"],
         "a react web dashboard")


# ── entry_mode: an allow-list, never a deny-list ────────────────────────────────
def _mixed_modes(conn):
    """Two human-attested atoms, one on a fictional mode, one on the reserved 'frontier' mode.

    The last is the one that matters: it stands in for a mode added after any given filter was
    written. The fictional one is fictional on purpose — every real mode is either inside
    HUMAN_ATTESTED or is `frontier`, so naming a real one here would pin that mode's status
    instead of the allow-list discipline."""
    _add(conn, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools", entry_mode="user-saved")
    _add(conn, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", entry_mode="oracle-footprint")
    _add(conn, C, "github", "artifact", "github:root", ["ai-agents"],
         "an agent framework toolkit", entry_mode="not-a-real-mode")
    _add(conn, D, "github", "artifact", "github:root", ["ai-agents"],
         "an agent framework runtime", entry_mode="frontier")


def test_entry_mode_allow_lists_the_named_modes(kb_home):
    conn = schema.connect()
    _mixed_modes(conn)
    hits = search_atoms(conn, "agent framework",
                        entry_mode=list(schema.HUMAN_ATTESTED), k=8).hits
    assert {h.atom_id for h in hits} == {A, B}
    conn.close()


def test_entry_mode_excludes_a_new_mode_by_default(kb_home):
    """A mode nobody named must not arrive on its own. If this ever fails, the filter
    has become a deny-list and a stage-3 admission reaches the default scope."""
    conn = schema.connect()
    _mixed_modes(conn)
    hits = search_atoms(conn, "agent framework",
                        entry_mode=list(schema.HUMAN_ATTESTED), k=8).hits
    assert D not in {h.atom_id for h in hits}      # 'frontier' was never named
    conn.close()


def test_entry_mode_addresses_the_autonomous_pile(kb_home):
    """The other half of why this exists: the pile stage 2's re-rank has to be able to demote.
    Selecting TO the non-human modes must work as well as selecting away from them."""
    conn = schema.connect()
    _mixed_modes(conn)
    hits = search_atoms(conn, "agent framework",
                        entry_mode=["not-a-real-mode", "frontier"], k=8).hits
    assert {h.atom_id for h in hits} == {C, D}
    conn.close()


def test_empty_entry_mode_list_matches_nothing(kb_home):
    """Same rule as `who_id` and `tags`: a computed-empty set must not widen back to everything."""
    conn = schema.connect()
    _mixed_modes(conn)
    assert search_atoms(conn, "agent framework", entry_mode=[], k=8).hits == []
    assert candidate_atom_ids(conn, None, None, None, entry_mode=[]) == set()
    conn.close()


def test_entry_mode_is_priced_by_filter_costs(kb_home):
    """`filter_costs` says every filter `_filter_clauses` knows about must appear in its args, or
    the baseline is taken without it and every other filter's reported cost comes out wrong."""
    conn = schema.connect()
    _mixed_modes(conn)
    costs = filter_costs(conn, entry_mode=list(schema.HUMAN_ATTESTED))
    assert costs.get("entry_mode") == 2      # dropping it would reach C and D as well
    conn.close()


def test_tag_and_kind_prefilter_constrains_the_search(kb_home):
    conn = schema.connect()
    _corpus(conn)
    hits = search_atoms(conn, "framework",
                        tags=["ai-agents"], what_kind="artifact", k=8).hits
    ids = {h.atom_id for h in hits}
    assert ids == {A, B}                 # crypto opinion + web repo are filtered OUT
    conn.close()


def test_tag_filter_with_no_match_returns_empty(kb_home):
    conn = schema.connect()
    _corpus(conn)
    assert search_atoms(conn, "framework", tags=["does-not-exist"]).hits == []
    conn.close()


# ── who_name: a display label, not a query surface ──────────────────────────────
# The atom stores `who_id` (`x:user:33836629`, `github:karpathy`) — an opaque token. The display
# name lives on `entities` and is joined in to label every hit. These pin what the join does
# (label every hit) and what it does not do (make authorship filterable).

def test_hits_carry_the_author_display_name(kb_home):
    conn = schema.connect()
    _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")

    hits = search_atoms(conn, "framework", tags=["ai-agents"], k=8).hits
    named = {h.atom_id: h.who_name for h in hits}
    assert named[A] == "Root McRepo"        # A's author is github:root
    conn.close()


def test_a_missing_entity_row_still_returns_the_hit(kb_home):
    """LEFT JOIN, not INNER. `_corpus` writes atoms without ever creating entity rows, so an
    inner join would silently drop every atom from retrieval. The name is a label; losing a
    label must never cost the result."""
    conn = schema.connect()
    _corpus(conn)            # no upsert_entity anywhere

    hits = search_atoms(conn, "framework", tags=["ai-agents"], k=8).hits
    assert {h.atom_id for h in hits} == {A, B}      # still returned...
    assert all(h.who_name is None for h in hits)     # ...just unnamed
    conn.close()


def test_who_id_remains_the_identity(kb_home):
    """The name is decoration. Two people can share a display name; `who_id` cannot collide."""
    conn = schema.connect()
    _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Same Name")
    schema.upsert_entity(conn, "github:stranger", name="Same Name")

    hits = search_atoms(conn, "framework", tags=["ai-agents"], k=8).hits
    assert {h.who_name for h in hits} == {"Same Name"}          # indistinguishable by name
    assert {h.who_id for h in hits} == {"github:root", "github:stranger"}   # distinct by id
    conn.close()


# ── who_id: authorship as a filter (the id, never the name) ─────────────────────

def test_who_id_filter_returns_only_that_authors_atoms(kb_home):
    """github:root wrote A and D;
    a `who_id` filter must return those and nothing else, regardless of relevance."""
    conn = schema.connect()
    _corpus(conn)

    hits = search_atoms(conn, "framework dashboard",
                        who_id="github:root", k=8).hits
    assert {h.atom_id for h in hits} == {A, D}
    assert all(h.who_id == "github:root" for h in hits)
    conn.close()


def test_unknown_who_id_returns_nothing_not_everything(kb_home):
    """A filter that silently falls back to the whole store is worse than no filter: the
    caller believes it scoped the search and it did not. Empty is the honest answer."""
    conn = schema.connect()
    _corpus(conn)
    assert search_atoms(conn, "framework", who_id="x:user:nobody").hits == []
    conn.close()


def test_who_id_ANDs_with_the_other_filters(kb_home):
    """Every pre-filter clause narrows; none replaces another. github:root wrote A (ai-agents)
    and D (web), so adding a tag must cut it to one."""
    conn = schema.connect()
    _corpus(conn)

    hits = search_atoms(conn, "framework dashboard",
                        who_id="github:root", tags=["web"], k=8).hits
    assert {h.atom_id for h in hits} == {D}
    conn.close()


def test_the_filter_beats_the_name_in_the_query(kb_home):
    """Why the filter exists, stated as a test.

    A name in the query matches by content, so an atom that merely mentions someone ranks
    alongside one they wrote — on a real 40-atom store, the query
    "Anthropic" returned 5 hits of which 1 was theirs and 4 only mentioned them. Here C is an
    x-opinion by someone else whose text contains "agents"; the naive query cannot separate it
    from github:root's own work, and the filter can.
    """
    conn = schema.connect()
    _corpus(conn)
    _add(conn, "x:2", "x", "opinion", "x:user:9", [],
         "github:root builds an autonomous agent framework worth reading")

    naive = {h.atom_id for h in search_atoms(conn, "agent framework",
                                             k=8).hits}
    assert "x:2" in naive, "the mention ranks alongside the real thing — the defect"

    scoped = {h.atom_id for h in search_atoms(conn, "agent framework",
                                              who_id="github:root", k=8).hits}
    assert "x:2" not in scoped and A in scoped
    conn.close()


def test_name_resolution_is_deliberately_NOT_here(kb_home):
    """The filter takes an id, never a display name, on purpose.

    Names collide, are user-controlled (a live sample carried 29 zero-width characters), and
    resolving one would have to choose whether to follow `entities.canonical_id`. Passing a
    name must therefore miss rather than quietly guess an author.
    """
    conn = schema.connect()
    _corpus(conn)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")

    assert search_atoms(conn, "framework", who_id="Root McRepo").hits == []
    conn.close()


# ── `tags` matches `payload.source_tags` ─────────────────────────────────────────
# `payload.source_tags` (the author's own declared labels) is the only tag space the filter
# matches.

def test_tags_match_author_declared_source_tags(kb_home):
    """An atom's author-declared labels, in `payload.source_tags`, are what `tags` matches."""
    conn = schema.connect()
    _corpus(conn)
    _add(conn, "github:root/tokio", "github", "artifact", "github:root",
         [], "an async runtime framework for writing network applications",
         source_tags=["rust", "async"])

    hits = search_atoms(conn, "framework", tags=["rust"], k=8).hits
    assert {h.atom_id for h in hits} == {"github:root/tokio"}
    conn.close()


def test_one_tag_list_matches_across_different_tag_values(kb_home):
    """A single `tags` call ORs across every requested value — the host asks for several
    labels at once and gets every atom carrying any one of them."""
    conn = schema.connect()
    _corpus(conn)
    _add(conn, "github:root/tokio", "github", "artifact", "github:root",
         [], "an async runtime framework", source_tags=["rust"])

    hits = search_atoms(conn, "framework", tags=["ai-agents", "rust"], k=8).hits
    assert {h.atom_id for h in hits} == {A, B, "github:root/tokio"}
    conn.close()


def test_tags_that_normalize_away_match_nothing_not_everything(kb_home):
    """The silent-widen twin of `test_unknown_who_id_returns_nothing_not_everything`.

    `slugify("+++")` is `""`, so a `tags=["+++"]` request survives normalization as an empty
    list. Read as "no tag clause", the
    query has no filters left at all and comes back with the whole store, presented to a caller
    who believes it was tag-restricted. Empty is the honest answer."""
    conn = schema.connect()
    _corpus(conn)

    assert candidate_atom_ids(conn, [], None, None, None) == set()
    # And through the real entry point, where `slug_tags` does the normalizing.
    assert kb_entry.run_kb_search("framework", tags=["+++"], k=8)["hits"] == []
    conn.close()


def test_no_tags_at_all_still_means_no_filter(kb_home):
    """The other empty case, reading the opposite way on purpose — the same asymmetry the
    `who_id` pair draws. Never asking for a tag filter is not the same as asking for one that
    matched nothing, so `None` must stay "search the whole store"."""
    conn = schema.connect()
    _corpus(conn)
    assert candidate_atom_ids(conn, None, None, None, None) is None
    conn.close()


def test_ranking_is_pure_relevance_not_trust(kb_home):
    """Ranking is pure relevance — no author-identity signal re-ranks it. B is a tighter keyword match for
    'agent framework' (fewer stray words → higher BM25), so B leads even though A's author is
    the one the user endorsed. Nothing about WHO wrote it may float A above the closer match."""
    conn = schema.connect()
    _corpus(conn)

    hits = search_atoms(conn, "agent framework",
                        tags=["ai-agents"]).hits
    assert hits[0].atom_id == B                          # tighter match wins; identity doesn't override
    assert {h.atom_id for h in hits} == {A, B}
    conn.close()


def test_a_hit_carries_its_excerpt_and_where_it_sits(kb_home):
    conn = schema.connect()
    _corpus(conn)
    top = search_atoms(conn, "agent framework", tags=["ai-agents"]).hits[0]
    start, end = top.snippet_span
    assert top.snippet and schema.read_body(conn, top.atom_id)[start:end] == top.snippet
    conn.close()


def test_bm25_arm_finds_literal_token(kb_home):
    conn = schema.connect()
    _corpus(conn)
    hits = search_atoms(conn, "framework").hits
    ids = {h.atom_id for h in hits}
    assert A in ids and B in ids          # both contain the literal "framework"
    assert C not in ids and D not in ids  # neither crypto nor the dashboard mention it
    conn.close()



# ── a match deep in a long item ─────────────────────────────────────────────────
# An item is indexed and ranked whole, so nothing
# but the index decides whether a paragraph buried in a long document answers a query, and the
# excerpt must be cut where the index matched rather than at the document's opening.
LONG = "paper:long/deep"      # 8 sections; its only match sits deep, in section 5 of 8
SHORT = "x:short/tight"       # an exact match — the control


def _window(marker: str) -> str:
    """One 1400-char section of snapshot: neutral filler with `marker` buried ~400 chars in.
    The filler ('nn'/'x') holds no query word, so the marker is the only thing that can match."""
    return ("nn " * 134 + marker + " ").ljust(1400, "x")[:1400]


def _deep_corpus(conn):
    _add(conn, LONG, "paper", "artifact", "openalex:x", [],
         "".join(_window("agent framework library" if i == 5 else "crypto") for i in range(8)))
    _add(conn, SHORT, "x", "opinion", "x:user:2", [], "agent framework")


def test_a_deep_span_in_a_long_atom_is_findable(kb_home):
    """LONG's only match sits deep in its body — no title, no head — and it is still returned,
    with the excerpt on the span that matched rather than on the document's beginning."""
    conn = schema.connect()
    _deep_corpus(conn)
    hits = search_atoms(conn, "agent framework", k=8).hits
    deep = next(h for h in hits if h.atom_id == LONG)
    assert deep.snippet_span[0] > 0                  # the deep window, not the opening
    assert "agent framework" in deep.snippet
    conn.close()

