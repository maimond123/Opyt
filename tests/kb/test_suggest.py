"""What a store can support, measured — and every way that measurement could lie quietly.

The failures this locks are all SILENT ones. A suggestion is a plausible sentence about the
user's own material, and nothing downstream can check a count it was handed: if the author total
is wrong the host repeats it, if the lens is wrong the user pays for a reading that does not work,
and in neither case does anything raise. So each test below fixes a number that was, or could be,
confidently wrong.

  • A ONE-VOICE STORE NEVER GETS `disconfirmation`. One author cannot contradict themselves into
    a finding, and `lens_warnings`' single-author rule is scoped to `queries` — so nothing else
    catches it.
  • A THIN STORE LEADS WITH `richer`, not with "go find more people". Naming a subject is the one
    move where OPYT does the work and the user does not, and it is worth most in the store that
    can do least. "Collect more people" is homework, and it was the only thing on offer for years.
  • `wider` IS ALWAYS PRESENT AND ALWAYS LAST. More people is a legitimate want; removing it would
    be the same mistake pointed the other way.
  • THE INVENTORY COUNTS THE WHOLE STORE. Both first attempts under-reported it plausibly.
  • A PEER'S AGGREGATE GETS NO SUGGESTIONS. Advice to read atoms the reader does not hold.
"""
from __future__ import annotations

import json

import pytest

from opyt_core.kb import kb_aggregate
from opyt_core.suggest import suggestions
from pipeline.kb import schema
from pipeline.kb import sitting_surface as ss


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _seed(conn, rows):
    """rows: (who_id, when_ts, tag) — the three columns every measurement here reads."""
    for i, (who, when, tag) in enumerate(rows):
        conn.execute(
            "INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode, what_kind, "
            "description, payload) VALUES (?,?,?,?,'user-saved','post',?,?)",
            (f"a:{i}", "substack", who, when, f"post {i}",
             json.dumps({"source_tags": [tag]})))
    conn.commit()


def _month(n):
    return f"2026-{n:02d}-05"


# ── picking the reading that fits ───────────────────────────────────────────────
def test_a_single_author_store_is_offered_a_trajectory_and_never_a_disconfirmation(conn):
    """MEASURED SHAPE, NOT A GUESS AT INTENT. 88 posts by one person over a year is the best
    input `sitting` has for how a mind CHANGED, and the worst for what contradicts what. The old
    fixed menu offered "build a focused reading session" either way."""
    _seed(conn, [("x:taelin", _month(1 + i % 12), "post-training") for i in range(88)])

    block = suggestions(conn, kb_aggregate())
    call = block["choices"][0]["call"]

    assert "lens='trajectory'" in call
    assert "disconfirmation" not in json.dumps(block)


def test_a_crowded_topic_is_offered_a_briefing(conn):
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    assert "lens='briefing'" in suggestions(conn, kb_aggregate())["choices"][0]["call"]


# ── the floor ───────────────────────────────────────────────────────────────────
def test_a_thin_store_leads_with_watching_a_subject_not_with_finding_more_people(conn):
    """⚠️ THE REBALANCE, IN ONE ASSERTION. A four-atom topic cannot be read end to end, and the
    answer to that used to be a list of ways to go find more writers — homework, handed to the
    user at the exact moment they have least reason to trust the thing. Watching the subject is
    the move where OPYT does the work instead, it needs no sitting and no spend, and it was built
    and never offered."""
    _seed(conn, [("x:a", _month(8), "agents"), ("x:b", _month(9), "agents"),
                 ("x:a", _month(9), "rag"), ("x:b", _month(9), "rag")])

    block = suggestions(conn, kb_aggregate())

    assert [c["direction"] for c in block["choices"]] == ["richer", "wider"]
    assert "watchlist" in block["choices"][0]["call"]
    assert "standalone starts at 10" in block["not_yet"]["sitting"]
    assert "onboard(source=" not in json.dumps(block)


def test_a_store_deep_enough_to_read_leads_with_deeper(conn):
    """The other side of the same rule: where the material CAN be read, reading it wins. Watching
    a subject is what you do when there is not enough yet, not a replacement for the material."""
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    assert suggestions(conn, kb_aggregate())["choices"][0]["direction"] == "deeper"


def test_finding_more_people_is_offered_last_and_never_dropped(conn):
    """Demoted, not deleted. It was the only thing on offer for years; removing it would be the
    same mistake pointed the other way."""
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    assert suggestions(conn, kb_aggregate())["choices"][-1]["direction"] == "wider"


def test_richer_always_carries_the_survey_not_only_for_broad_subjects(conn):
    """The web survey is enrichment in its own right — papers, repos and authors the watch's
    three sources may miss — so it rides every RICHER offer, measured store included, not just
    the ones where the subject looks broad."""
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    richer = next(c for c in suggestions(conn, kb_aggregate())["choices"]
                  if c["direction"] == "richer")
    assert "web-search" in richer["note"]


def test_wider_has_a_no_names_fallback_that_keeps_the_finding_host_side(conn):
    """A user who picks WIDER but cannot name anyone used to hit a dead end — "give me names"
    is the question they picked this direction to avoid. The fallback is the host finding
    people from the store's own topics, brought back as candidates to CONFIRM: web-found
    people are the weakest credibility signal of any discovery path here, so the screen gate
    is not optional for them."""
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    wider = next(c for c in suggestions(conn, kb_aggregate())["choices"]
                 if c["direction"] == "wider")
    assert "no names" in wider["why"]
    assert "never auto-added" in wider["why"]


def test_a_subject_already_being_watched_is_named_so_it_is_not_offered_again(conn):
    """An offer the user accepts that then changes nothing spends their turn to tell them so.

    ⚠️ THE SAME PRINCIPLE, MOVED ONE STEP LATER (2026-09-16). This used to drop RICHER entirely
    when the store's top `source_tags` value was already watched. Both halves of that are gone:
    RICHER no longer proposes a subject at all — the user names one — so there is nothing to
    suppress at offer time, and the tag it suppressed against was never the store's subject in
    the first place (5 tagged atoms of 1,801 on the live store).

    So the brake now applies where the subject actually exists: the offer carries what is
    already on watch, and the host does not re-propose it. Asserting the subject is NAMED is
    what keeps this a real brake rather than a deleted one — a RICHER offer that stayed silent
    about an existing watch would let the host walk the user straight back into the no-op.
    """
    from pipeline.kb import frontier_queries as fq
    _seed(conn, [("x:a", _month(8), "agents") for _ in range(4)])
    fq.add_user_query(conn, "agents")

    richer = next(c for c in suggestions(conn, kb_aggregate())["choices"]
                  if c["direction"] == "richer")
    assert "agents" in richer["note"]
    assert "do not propose these again" in richer["note"]


def test_a_large_store_is_never_told_it_is_too_thin_to_read(conn):
    """⚠️ REGRESSION, and the exact failure this file exists for: a plausible sentence about the
    user's own material that is flatly untrue and that nothing downstream can check.

    DEEPER used to be gated on `best_lens(shape) is not None`. That held only while `shape` was
    measured over the top `source_tags` value, where the measured set roughly WAS a region. Once
    the tag seed was removed (2026-09-16) and the shape became the whole store, every rule in
    `lens_warnings` — written to describe a BUILT REGION — started applying to a store, and
    several of them fire for reasons that have nothing to do with having too little material.

    One undated atom is enough. On the live store that was 1 atom of 1,801; it spoils every
    lens, DEEPER vanished, and RICHER told the user *"You have only 1801 items here, which is
    too few to read end to end"*. The fixture below is that store in miniature: plenty of
    material, one atom with no date. Thinness is now asked of the reading tier, which is the
    only property that means it.
    """
    _seed(conn, [(f"x:w{i % 5}", _month(1 + i % 9), "agents") for i in range(40)])
    conn.execute("UPDATE atoms SET when_ts = NULL WHERE atom_id = 'a:0'")
    conn.commit()

    block = suggestions(conn, kb_aggregate())

    assert block["choices"][0]["direction"] == "deeper"
    assert "too few" not in json.dumps(block)
    assert "not_yet" not in block
    # The lens is dropped rather than asserted: no lens fits this shape cleanly, and naming one
    # anyway would trade a false claim about SIZE for a false claim about FIT.
    assert block["choices"][0]["call"] == "sitting(query='…')"


def test_richer_says_nothing_about_watches_when_there_are_none(conn):
    """The other half: a store watching nothing must not carry an empty do-not-propose list.

    These strings are instructions to a model that expands them, so a trailing "Already on
    watch: ." is not cosmetic — it is a sentence about the user's setup that the host will try
    to make sense of and may well repeat."""
    _seed(conn, [("x:a", _month(8), "agents") for _ in range(4)])

    richer = next(c for c in suggestions(conn, kb_aggregate())["choices"]
                  if c["direction"] == "richer")
    assert "Already on watch" not in richer["note"]


def test_an_empty_store_suggests_nothing_rather_than_reporting_emptiness(conn):
    """Fail-safe, and specifically the no-claim half of it: a store with nothing in it must not
    produce a sentence about what it has."""
    assert suggestions(conn, kb_aggregate()) == {}


def test_an_unreadable_store_degrades_to_silence(conn):
    """A suggestion is the most droppable thing in any response carrying one — it must never be
    the reason a real result fails to return."""
    _seed(conn, [("x:a", _month(8), "agents") for _ in range(12)])
    conn.close()                      # every query below now raises

    assert suggestions(conn, {"total": 12}) == {}


# ── orienting before asking ─────────────────────────────────────────────────────
def test_a_store_bigger_than_its_envelope_is_told_to_census_it_first(conn):
    """⚠️ THE HOLE THE TAG REMOVAL OPENED, AND HOW IT CLOSES. DEEPER and RICHER both need a
    subject, and since 2026-09-16 neither proposes one — the tag that used to fake it is gone
    and nothing in the retrieval stack can suggest a subject (`search` and `sitting` both take
    a query; the index verifies, it never proposes). So the host is told to go and look, with
    a call that shows breadth, before it puts the question."""
    _seed(conn, [(f"x:w{i % 9}", _month(1 + i % 9), f"t{i % 5}") for i in range(60)])

    block = suggestions(conn, kb_aggregate())

    assert block["orient"]["call"] == "aggregate(sample=200)"
    assert "12 of 60" in block["orient"]["why"]
    assert "CHECK each with `search(query=…)`" in block["orient"]["note"]
    assert "orient" in block["cap"], "the cap must stop `orient` being read out as a choice"
    # ⚠️ POSITION IS THE INSTRUCTION. `orient` is work the host does BEFORE putting the
    # question, and `cap` is the instruction to put it. Emitted after `cap` — which is what
    # happened until 2026-09-16 — a host reading top to bottom meets "ask now" first, and on
    # the live run it did: the census only ran because the user asked for it by hand.
    assert list(block) == ["orient", "have", "cap", "choices"]


def test_a_store_the_host_can_already_see_whole_is_not_told_to_census_it(conn):
    """Derived, not a threshold: the aggregate carries `recent_descriptions`, so a store no
    bigger than that list is one the host is already looking at in full. Telling it to go
    sample the thing in front of it spends a call to re-read what it has."""
    _seed(conn, [(f"x:w{i}", _month(1 + i), "agents") for i in range(6)])

    assert "orient" not in suggestions(conn, kb_aggregate())


# ── the inventory line ──────────────────────────────────────────────────────────
def test_the_inventory_counts_every_author_not_the_aggregate_top_fifteen(conn):
    """REGRESSION. `top_entities` is LIMIT 15, so reading the author count off it reported
    exactly fifteen for every larger store — a plausible number, silently wrong."""
    _seed(conn, [(f"x:w{i % 23}", _month(1 + i % 9), f"topic-{i % 9}") for i in range(200)])

    assert "23 people" in suggestions(conn, kb_aggregate())["have"]


def test_the_inventory_spans_the_store_not_its_busiest_topic(conn):
    """REGRESSION. Using the top topic's shape for the store's line said a store spanning
    January to September spanned only the month its busiest tag happened to sit in."""
    _seed(conn, [("x:a", _month(1), "quiet")]
                + [(f"x:w{i % 4}", _month(9), "busy") for i in range(20)])

    assert "2026-01–2026-09" in suggestions(conn, kb_aggregate())["have"]


# ── the cap ─────────────────────────────────────────────────────────────────────
def test_three_directions_is_the_ceiling(conn):
    """The cap is in the text for verbose models AND enforced here, because a model that ignores
    the sentence still cannot expand a choice that was never sent. Three is the ceiling because
    there are three directions; a fourth would mean a new KIND of thing to want, not another
    tool."""
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])
    conn.executemany("INSERT INTO frontier_candidates (candidate_id, source, status, first_seen_at, "
                     "last_seen_at) VALUES (?, 'arxiv', 'new', '2026-09-01', '2026-09-01')",
                     [(f"c:{i}",) for i in range(17)])
    conn.commit()

    block = suggestions(conn, kb_aggregate())

    assert len(block["choices"]) == 3
    assert "ONE short question" in block["cap"]


# ── the shape measurement itself ────────────────────────────────────────────────
def test_the_store_shape_reuses_sittings_own_thresholds_rather_than_its_own(conn):
    """The whole reason `store_shape` feeds `lens_warnings`: the moment a suggester carries its
    own idea of "too thin", it drifts from what `sitting` tells the user one call later.

    Measured over the WHOLE store since 2026-09-16 — this was `shape_by_tag(conn, "agents")`
    back when `suggest` measured the top `source_tags` value. The fixture is unchanged because
    the seeded store IS the four atoms, so the number under test is the same one; what moved is
    that it no longer arrives via a tag nobody chose.
    """
    _seed(conn, [("x:a", _month(8), "agents") for _ in range(4)])

    shape = ss.store_shape(conn)
    lens, missing = ss.best_lens(shape)

    assert lens is None and shape["tier"] == "sprout"
    assert missing == ss.lens_warnings(shape, "briefing")


# ── whose store the advice is about ─────────────────────────────────────────────
def test_a_peers_aggregate_gets_counts_and_no_advice(conn):
    """`aggregate(kb=...)` summarizes somebody ELSE'S store, and every suggestion here tells the
    READER what to do next. "Read these 30 in order" is true of the peer's material and impossible
    for the reader, who does not hold a single one of those atoms."""
    from mcp_server.atoms_tools import _attach_suggestions
    _seed(conn, [(f"x:w{i % 6}", _month(2 + i % 7), "agentic-payments") for i in range(30)])

    mine, theirs = kb_aggregate(), kb_aggregate()
    _attach_suggestions(mine, None)
    _attach_suggestions(theirs, "will")

    assert "suggested" in mine
    assert "suggested" not in theirs
