"""retrieve.search_atoms — the enforced-hybrid mechanics, proven offline with the
deterministic bag-of-words embedder: the tag pre-filter (over `payload.source_tags`), the
BM25 arm, and the max-pooled semantic arm.

Ranking is PURE RELEVANCE. The trust re-rank this docstring used to describe was removed
2026-07-15, and `test_ranking_is_pure_relevance_not_trust` below is the assertion of that.
The trust-tier layer it re-ranked on was itself deleted 2026-08-23, having sat callerless
in between — see docs/plans/2026-08-23-delete-edges-and-trust-tiers.md.
"""
from __future__ import annotations

from opyt_core import kb as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import candidate_atom_ids, filter_costs, search_atoms

# ── corpus ──────────────────────────────────────────────────────────────────────
A = "github:root/agentkit"   # ai-agents, rich text
B = "github:stranger/agents"  # ai-agents, terse text — the TIGHTER match
C = "x:1"                      # crypto opinion
D = "github:root/dash"        # web


def _add(conn, emb, atom_id, source_type, what_kind, who_id, topics, snapshot,
         source_tags=None, entry_mode="user-saved"):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, snapshot)
    tags = list(topics) + list(source_tags or [])
    atom = dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts="2024-05-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card",
        # `source_tags` ABSENT unless there is one — the realistic shape, since most sources
        # declare no labels. The absent key is what the filter must tolerate (json_each over
        # a missing key yields zero rows rather than erroring).
        payload={"source_tags": tags} if tags else {},
        entry_mode=entry_mode,
    )
    store_atom(conn, emb, atom=atom, snapshot_text=snapshot,
               )


def _corpus(conn, emb):
    _add(conn, emb, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents")
    _add(conn, emb, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library")
    _add(conn, emb, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems")
    _add(conn, emb, D, "github", "artifact", "github:root", ["web"],
         "a react web dashboard")


# ── entry_mode: an ALLOW-list, never a deny-list ────────────────────────────────
def _mixed_modes(conn, emb):
    """Two human-attested atoms, one on a FICTIONAL mode, one on the RESERVED 'frontier' mode.

    The last is the one that matters: it stands in for a mode added after any given filter was
    written. The fictional one is fictional on purpose — every real mode is now either inside
    HUMAN_ATTESTED or is `frontier`, so naming a real one here would pin that mode's status
    instead of the allow-list discipline."""
    _add(conn, emb, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools", entry_mode="user-saved")
    _add(conn, emb, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", entry_mode="oracle-footprint")
    _add(conn, emb, C, "github", "artifact", "github:root", ["ai-agents"],
         "an agent framework toolkit", entry_mode="not-a-real-mode")
    _add(conn, emb, D, "github", "artifact", "github:root", ["ai-agents"],
         "an agent framework runtime", entry_mode="frontier")


def test_entry_mode_allow_lists_the_named_modes(kb_home, fake_embedder):
    conn = schema.connect()
    _mixed_modes(conn, fake_embedder)
    hits = search_atoms(conn, "agent framework", fake_embedder,
                        entry_mode=list(schema.HUMAN_ATTESTED), k=8).hits
    assert {h.atom_id for h in hits} == {A, B}
    conn.close()


def test_entry_mode_excludes_a_new_mode_by_default(kb_home, fake_embedder):
    """THE INVARIANT. A mode nobody named must not arrive on its own. If this ever fails, the filter
    has become a deny-list and a stage-3 admission can re-enter the generator."""
    conn = schema.connect()
    _mixed_modes(conn, fake_embedder)
    hits = search_atoms(conn, "agent framework", fake_embedder,
                        entry_mode=list(schema.HUMAN_ATTESTED), k=8).hits
    assert D not in {h.atom_id for h in hits}      # 'frontier' was never named
    conn.close()


def test_entry_mode_addresses_the_autonomous_pile(kb_home, fake_embedder):
    """The other half of why this exists: the pile stage 2's re-rank has to be able to demote.
    Selecting TO the non-human modes must work as well as selecting away from them."""
    conn = schema.connect()
    _mixed_modes(conn, fake_embedder)
    hits = search_atoms(conn, "agent framework", fake_embedder,
                        entry_mode=["not-a-real-mode", "frontier"], k=8).hits
    assert {h.atom_id for h in hits} == {C, D}
    conn.close()


def test_empty_entry_mode_list_matches_nothing(kb_home, fake_embedder):
    """Same rule as `who_id` and `tags`: a computed-empty set must not widen back to everything."""
    conn = schema.connect()
    _mixed_modes(conn, fake_embedder)
    assert search_atoms(conn, "agent framework", fake_embedder, entry_mode=[], k=8).hits == []
    assert candidate_atom_ids(conn, None, None, None, entry_mode=[]) == set()
    conn.close()


def test_entry_mode_is_priced_by_filter_costs(kb_home, fake_embedder):
    """`filter_costs` says every filter `_filter_clauses` knows about must appear in its args, or
    the BASELINE is taken without it and every OTHER filter's reported cost comes out wrong."""
    conn = schema.connect()
    _mixed_modes(conn, fake_embedder)
    costs = filter_costs(conn, entry_mode=list(schema.HUMAN_ATTESTED))
    assert costs.get("entry_mode") == 2      # dropping it would reach C and D as well
    conn.close()


def test_human_attested_has_one_spelling(kb_home):
    """The constant moved to `schema.py` so the retriever and the sitting rail cannot drift."""
    from pipeline.kb import sitting_vectors as sv
    assert sv.HUMAN_ATTESTED is schema.HUMAN_ATTESTED


def test_tag_and_kind_prefilter_constrains_both_arms(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    hits = search_atoms(conn, "framework", fake_embedder,
                        tags=["ai-agents"], what_kind="artifact", k=8, mode="hybrid").hits
    ids = {h.atom_id for h in hits}
    assert ids == {A, B}                 # crypto opinion + web repo are filtered OUT
    conn.close()


def test_tag_filter_with_no_match_returns_empty(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    assert search_atoms(conn, "framework", fake_embedder, tags=["does-not-exist"]).hits == []
    conn.close()


# ── who_name: a display LABEL, not a query surface ──────────────────────────────
# The atom stores `who_id` (`x:user:33836629`, `github:karpathy`) — an opaque token. The display
# name lives on `entities`, and until now retrieval never joined it, so a host got the token and
# nothing else. These pin what the join DOES (label every hit) and, just as importantly, what it
# does NOT do (make authorship filterable).

def test_hits_carry_the_author_display_name(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")

    hits = search_atoms(conn, "framework", fake_embedder, tags=["ai-agents"], k=8).hits
    named = {h.atom_id: h.who_name for h in hits}
    assert named[A] == "Root McRepo"        # A's author is github:root
    conn.close()


def test_a_missing_entity_row_still_returns_the_hit(kb_home, fake_embedder):
    """LEFT JOIN, not INNER. `_corpus` writes atoms without ever creating entity rows, so an
    INNER join would silently drop EVERY atom from retrieval — a catastrophic, invisible
    regression. The name is a label; losing a label must never cost the result."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)            # no upsert_entity anywhere

    hits = search_atoms(conn, "framework", fake_embedder, tags=["ai-agents"], k=8).hits
    assert {h.atom_id for h in hits} == {A, B}      # still returned...
    assert all(h.who_name is None for h in hits)     # ...just unnamed
    conn.close()


def test_who_id_remains_the_identity(kb_home, fake_embedder):
    """The name is decoration. Two people can share a display name; `who_id` cannot collide."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Same Name")
    schema.upsert_entity(conn, "github:stranger", name="Same Name")

    hits = search_atoms(conn, "framework", fake_embedder, tags=["ai-agents"], k=8).hits
    assert {h.who_name for h in hits} == {"Same Name"}          # indistinguishable by name
    assert {h.who_id for h in hits} == {"github:root", "github:stranger"}   # distinct by id
    conn.close()


# ── who_id: authorship as a FILTER (the id, never the name) ─────────────────────

def test_who_id_filter_returns_only_that_authors_atoms(kb_home, fake_embedder):
    """The capability the display-name join did NOT provide. github:root wrote A and D;
    a `who_id` filter must return those and nothing else, regardless of relevance."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)

    hits = search_atoms(conn, "framework dashboard", fake_embedder,
                        who_id="github:root", k=8).hits
    assert {h.atom_id for h in hits} == {A, D}
    assert all(h.who_id == "github:root" for h in hits)
    conn.close()


def test_unknown_who_id_returns_nothing_not_everything(kb_home, fake_embedder):
    """A filter that silently falls back to the whole store is worse than no filter: the
    caller believes it scoped the search and it did not. Empty is the honest answer."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    assert search_atoms(conn, "framework", fake_embedder, who_id="x:user:nobody").hits == []
    conn.close()


def test_who_id_ANDs_with_the_other_filters(kb_home, fake_embedder):
    """Every pre-filter clause narrows; none replaces another. github:root wrote A (ai-agents)
    and D (web), so adding a tag must cut it to one."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)

    hits = search_atoms(conn, "framework dashboard", fake_embedder,
                        who_id="github:root", tags=["web"], k=8).hits
    assert {h.atom_id for h in hits} == {D}
    conn.close()


def test_the_filter_beats_the_name_in_the_query(kb_home, fake_embedder):
    """WHY the filter exists, stated as a test.

    A name in the QUERY matches by CONTENT, so an atom that merely MENTIONS someone ranks
    alongside one they WROTE — measured on a real 40-atom store 2026-08-03, the query
    "Anthropic" returned 5 hits of which 1 was theirs and 4 only mentioned them. Here C is an
    x-opinion by someone else whose text contains "agents"; the naive query cannot separate it
    from github:root's own work, and the filter can.
    """
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    _add(conn, fake_embedder, "x:2", "x", "opinion", "x:user:9", [],
         "github:root builds an autonomous agent framework worth reading")

    naive = {h.atom_id for h in search_atoms(conn, "agent framework", fake_embedder,
                                             k=8, mode="bm25").hits}
    assert "x:2" in naive, "the mention ranks alongside the real thing — the defect"

    scoped = {h.atom_id for h in search_atoms(conn, "agent framework", fake_embedder,
                                              who_id="github:root", k=8, mode="bm25").hits}
    assert "x:2" not in scoped and A in scoped
    conn.close()


def test_name_resolution_is_deliberately_NOT_here(kb_home, fake_embedder):
    """The filter takes an ID, never a display name — and that is a decision, not an omission.

    Names collide, are user-controlled (a live sample carried 29 zero-width characters), and
    resolving one would have to choose whether to follow `entities.canonical_id`. Passing a
    name must therefore MISS rather than quietly guess an author.
    """
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    schema.upsert_entity(conn, "github:root", name="Root McRepo")

    assert search_atoms(conn, "framework", fake_embedder, who_id="Root McRepo").hits == []
    conn.close()


# ── `tags` matches `payload.source_tags` ─────────────────────────────────────────
# `about_topics` (a machine-derived slug space meant to be backfilled by Stage-6 clustering)
# was retired 2026-08-17, never having been populated — see
# docs/plans/2026-08-17-about-topics-column-retired.md. `payload.source_tags` (the author's
# own declared labels) is the only tag space the filter matches today.

def test_tags_match_author_declared_source_tags(kb_home, fake_embedder):
    """An atom's author-declared labels, in `payload.source_tags`, are what `tags` matches."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    _add(conn, fake_embedder, "github:root/tokio", "github", "artifact", "github:root",
         [], "an async runtime framework for writing network applications",
         source_tags=["rust", "async"])

    hits = search_atoms(conn, "framework", fake_embedder, tags=["rust"], k=8).hits
    assert {h.atom_id for h in hits} == {"github:root/tokio"}
    conn.close()


def test_one_tag_list_matches_across_different_tag_values(kb_home, fake_embedder):
    """A single `tags` call ORs across every requested value — the host asks for several
    labels at once and gets every atom carrying any one of them."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    _add(conn, fake_embedder, "github:root/tokio", "github", "artifact", "github:root",
         [], "an async runtime framework", source_tags=["rust"])

    hits = search_atoms(conn, "framework", fake_embedder, tags=["ai-agents", "rust"], k=8).hits
    assert {h.atom_id for h in hits} == {A, B, "github:root/tokio"}
    conn.close()


def test_tags_that_normalize_away_match_nothing_not_everything(kb_home, fake_embedder):
    """The silent-widen twin of `test_unknown_who_id_returns_nothing_not_everything`.

    `slugify("+++")` is `""`, so a `tags=["+++"]` request survives normalization as an EMPTY
    list. Read as "no tag clause" — which is what the pre-filter did until 2026-08-04 — the
    query has no filters left at all and comes back with the WHOLE STORE, presented to a caller
    who believes it was tag-restricted. Empty is the honest answer."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)

    assert candidate_atom_ids(conn, [], None, None, None) == set()
    # And through the real entry point, where `_slug_tags` does the normalizing.
    assert kb_entry.run_kb_search("framework", tags=["+++"], mode="bm25", k=8)["hits"] == []
    conn.close()


def test_no_tags_at_all_still_means_no_filter(kb_home, fake_embedder):
    """The other empty case, reading the opposite way on purpose — the same asymmetry the
    `who_id` pair draws. Never asking for a tag filter is not the same as asking for one that
    matched nothing, so `None` must stay "search the whole store"."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    assert candidate_atom_ids(conn, None, None, None, None) is None
    conn.close()


def test_ranking_is_pure_relevance_not_trust(kb_home, fake_embedder):
    """Ranking is pure relevance — no author-identity signal re-ranks it (the trust re-rank went
    2026-07-15, the tier layer it read went 2026-08-23). B is a TIGHTER bag-of-words match for
    'agent framework' (fewer stray words → higher cosine), so B leads even though A's author is
    the one the user endorsed. Nothing about WHO wrote it may float A above the closer match."""
    conn = schema.connect()
    _corpus(conn, fake_embedder)

    hits = search_atoms(conn, "agent framework", fake_embedder,
                        tags=["ai-agents"], mode="semantic").hits
    assert hits[0].atom_id == B                          # tighter match wins; identity doesn't override
    assert {h.atom_id for h in hits} == {A, B}
    conn.close()


def test_semantic_arm_reports_matched_span(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    hits = search_atoms(conn, "agent framework", fake_embedder, tags=["ai-agents"],
                        mode="semantic").hits
    top = hits[0]
    assert top.snippet                                   # the matched chunk text is carried
    assert top.chunk_span is not None and top.chunk_span[0] <= top.chunk_span[1]
    conn.close()


def test_bm25_arm_finds_literal_token(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    hits = search_atoms(conn, "framework", fake_embedder, mode="bm25").hits
    ids = {h.atom_id for h in hits}
    assert A in ids and B in ids          # both contain the literal "framework"
    assert C not in ids and D not in ids  # neither crypto nor the dashboard mention it
    conn.close()



# ── max-pool: what it buys (2026-08-27) ─────────────────────────────────────────
# An atom scores as its single best chunk. That is what lets a paragraph buried in a long
# document answer a query, and it is also why a long atom scores higher for its length — it
# draws its maximum from more samples (+0.066 for the SAME content chunked both ways, measured
# 2026-08-26). A percentile-within-chunk-band correction was built for that and REVERSED; the
# record is docs/plans/2026-08-26-frontier-crowding-in-search.md, and what contains the bias
# now is provenance sectioning in opyt_core/kb.py.
#
# The RESIDUAL is deliberately not pinned here. It is a property of drawing a maximum from a
# distribution, and `FakeEmbedder` is deterministic bag-of-words over an 11-word vocabulary —
# there is no distribution to draw from, so any test written for it here would pass for a
# different reason than the one it claims. Its evidence is a live-corpus measurement and lives
# where live measurements live: docs/baselines/rank_percentile_eval.py.
LONG = "paper:long/deep"      # 8 chunks; its ONLY match sits DEEP, in chunk 5 of 8
SHORT = "x:short/tight"       # 1 chunk, an exact match — the control


def _window(marker: str) -> str:
    """One 1400-char step of snapshot: neutral filler with `marker` buried past the overlap.

    The chunker steps `CHUNK_CHARS - CHUNK_OVERLAP` = 1400 chars and each window is 1600 wide, so
    window i holds step i whole plus the first 200 chars of step i+1. Putting the marker at char
    ~400 keeps it in exactly ONE chunk. The filler ('nn'/'x') contains no vocabulary substring,
    so a marker-free chunk scores zero against any query the FakeEmbedder can see.
    """
    return ("nn " * 134 + marker + " ").ljust(1400, "x")[:1400]


def _deep_corpus(conn, emb):
    """Query "agent framework" scores a chunk as |W n {agent, framework}| / sqrt(2|W|), so both
    cosines below are exact: LONG's best chunk is {agent, framework, library} = 0.816, SHORT's
    only chunk is {agent, framework} = 1.000."""
    _add(conn, emb, LONG, "paper", "artifact", "openalex:x", [],
         "".join(_window("agent framework library" if i == 5 else "crypto") for i in range(8)))
    _add(conn, emb, SHORT, "x", "opinion", "x:user:2", [], "agent framework")


def test_a_deep_span_in_a_long_atom_is_findable(kb_home, fake_embedder):
    """WHY max-pool exists, and the case any normalizer has to keep. LONG's only match sits in
    chunk 5 of 8 — no title, no head — and it is still returned, with the hit pointing at the
    span that actually matched rather than at the document's beginning."""
    conn = schema.connect()
    _deep_corpus(conn, fake_embedder)
    hits = search_atoms(conn, "agent framework", fake_embedder, mode="semantic", k=8).hits
    deep = next(h for h in hits if h.atom_id == LONG)
    assert deep.chunk_seq == 5                       # the DEEP chunk, not chunk 0
    assert "agent framework" in deep.snippet
    conn.close()

