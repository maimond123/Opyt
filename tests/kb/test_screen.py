"""Stage-4 Oracle SCREEN — ranking, the kind classifier (degrade-open), and the recommended/
see-all partition. Pure over entities + curation_signals; the LLM is monkeypatched (no network,
no key), so the classify LOGIC (batching, per-batch degrade-open, index alignment,
cache, idempotency) is proven offline."""
from __future__ import annotations

import json
import re

import pytest

from pipeline.kb import curation_state, resolve, schema, screen


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _person(conn, eid, *, name=None, links=None, profile=None):
    schema.upsert_entity(conn, eid, name=name, identity_links=links, profile=profile)


# ── ranking: group-by-canonical + Fork-1 sort ──────────────────────────────────

def test_rank_pools_signals_across_the_canonical_cluster(conn):
    # x:user:1 (X website → carol.substack.com) + substack:carol (home) MERGE in Stage 3, so a
    # follow on X and a subscribe on Substack must POOL into ONE candidate with distinct=2.
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="Carol Writes", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    schema.add_signal(conn, "x:user:1", "follow", "x")
    schema.add_signal(conn, "substack:carol", "subscribe", "substack", extra={"is_paid": None})

    cands = screen.rank_candidates(conn)
    assert len(cands) == 1, [c.canonical_id for c in cands]
    c = cands[0]
    assert c.distinct_signals == 2 and c.has_endorsement and c.corroborated
    assert {"x:user:1", "substack:carol"} == set(c.members)
    assert c.name == "Carol"                      # prefers the X row's name (Fork 1)


def test_repeated_truncated_following_walks_do_not_retire_candidates(conn):
    """Only an accepted following count may establish the retirement baseline."""
    for i in range(4):
        entity_id = f"x:user:{i}"
        _person(conn, entity_id, name=f"P{i}")
        schema.add_signal(conn, entity_id, "follow", "x")

    starts = ["2026-08-12T00:00:00+00:00", "2026-08-12T06:00:00+00:00",
              "2026-08-12T12:00:00+00:00", "2026-08-12T18:00:00+00:00"]
    for started_at, found in zip(starts, (4, 4, 1, 1)):
        curation_state.record_run(conn, "x_following", status="ok", found=found,
                                  stored_after=4, now=started_at, started_at=started_at)
    conn.execute("UPDATE curation_signals SET last_confirmed_at=? WHERE entity_id='x:user:0'",
                 (starts[-1],))
    conn.execute("UPDATE curation_signals SET last_confirmed_at=? WHERE entity_id!='x:user:0'",
                 (starts[0],))
    conn.commit()

    assert {c.canonical_id for c in screen.rank_candidates(conn)} == {
        "x:user:0", "x:user:1", "x:user:2", "x:user:3"}


def test_sort_key_is_endorsement_then_distinct_then_count(conn):
    # A: two distinct content signals (bookmark+like), NO endorsement, very high count.
    _person(conn, "x:user:A", name="A")
    schema.add_signal(conn, "x:user:A", "save", "x", count=12)
    schema.add_signal(conn, "x:user:A", "like", "x", count=30)
    # B: one endorsement signal (follow), lowest possible count.
    _person(conn, "x:user:B", name="B")
    schema.add_signal(conn, "x:user:B", "follow", "x")
    # C: one content signal only.
    _person(conn, "x:user:C", name="C")
    schema.add_signal(conn, "x:user:C", "like", "x", count=3)

    order = [c.canonical_id for c in screen.rank_candidates(conn)]
    # B wins on endorsement alone, beating A's two content signals and 42 content acts: a follow
    # is a PERSON-level act and no amount of reading outranks it. REVERSAL, David 2026-08-23 —
    # this assertion previously read [A, B, C] under "revealed preference over a passive follow".
    # Then A > C inside the content tier, on distinct signals.
    assert order == ["x:user:B", "x:user:A", "x:user:C"]


def test_variety_beats_volume_inside_the_content_tier(conn):
    # Deliberately preserved by the 2026-08-23 reversal, which changed ONLY the primary key.
    _person(conn, "x:user:MIX", name="Mix")
    schema.add_signal(conn, "x:user:MIX", "save", "x", count=1)
    schema.add_signal(conn, "x:user:MIX", "like", "x", count=1)
    _person(conn, "x:user:VOL", name="Vol")
    schema.add_signal(conn, "x:user:VOL", "save", "x", count=5)

    order = [c.canonical_id for c in screen.rank_candidates(conn)]
    assert order == ["x:user:MIX", "x:user:VOL"]


def test_bookmark_and_like_carry_identical_weight(conn):
    # Neither is an endorsement, so both fall to the same tiebreak and order purely by count.
    _person(conn, "x:user:S3", name="S3")
    schema.add_signal(conn, "x:user:S3", "save", "x", count=3)
    _person(conn, "x:user:L2", name="L2")
    schema.add_signal(conn, "x:user:L2", "like", "x", count=2)

    order = [c.canonical_id for c in screen.rank_candidates(conn)]
    assert order == ["x:user:S3", "x:user:L2"]


def test_reflect_degrades_when_paid_unknown(conn):
    _person(conn, "substack:x", name="X")
    schema.add_signal(conn, "substack:x", "subscribe", "substack", extra={"is_paid": None})
    c = screen.rank_candidates(conn)[0]
    assert screen.reflect(c) == "you subscribe"          # NOT "(paid)" when unknown
    schema.add_signal(conn, "substack:x", "subscribe", "substack", extra={"is_paid": True})
    c2 = screen.rank_candidates(conn)[0]
    assert "you subscribe (paid)" in screen.reflect(c2)


# ── classifier: batch / cache / degrade-open ───────────────────────────────────

_KIND_BY_PREFIX = {"PERSON": "person", "ORG": "org", "MEDIA": "media"}


def _patch_llm(monkeypatch, *, fail_on_call: int | None = None) -> dict:
    """Patch `llm_client` with a fake that answers from each PROMPT LINE's own name — `ORG-7` → org.
    Answering from the prompt, rather than from a fixed index→kind dict, is what makes a cross-batch
    index misalignment FAIL the test: a verdict written onto the wrong person no longer matches that
    person's own name. `fail_on_call=k` raises on the k-th call only, to prove per-batch isolation.
    Returns the shared call counter."""
    state = {"n": 0}

    def fake(role, *, system, user, **kw):
        state["n"] += 1
        if fail_on_call == state["n"]:
            raise RuntimeError("breaker open")
        out = {}
        for line in user.splitlines():
            m = re.match(r"^(\d+)\. ([A-Z]+)-\d+ ", line)
            if m:
                out[m.group(1)] = _KIND_BY_PREFIX[m.group(2)]
        return type("R", (), {"text": json.dumps(out)})()

    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr(llm_client, "call", fake)
    return state


def _seed_named(conn, n):
    """n follow-only candidates whose NAME declares the kind the classifier should return. Ids are
    zero-padded so the count tiebreak orders them predictably (rank == seed order)."""
    for i in range(n):
        eid = f"x:user:{i:04d}"
        _person(conn, eid, name=f'{("PERSON", "ORG", "MEDIA")[i % 3]}-{i}',
                profile={"bio": f"bio {i}"})
        schema.add_signal(conn, eid, "follow", "x")


def _seed_three(conn):
    for eid, nm in [("x:user:1", "Person One"), ("x:user:2", "OpenAI"), ("x:user:3", "Person Three")]:
        _person(conn, eid, name=nm, profile={"bio": nm})
        schema.add_signal(conn, eid, "follow", "x")


def test_classify_assigns_caches_and_is_idempotent(conn, monkeypatch):
    _seed_three(conn)
    cands = screen.rank_candidates(conn)

    calls = {"n": 0}

    class _Resp:
        text = '{"1":"person","2":"org","3":"person"}'

    def fake_call(role, *, system, user, **kw):
        calls["n"] += 1
        return _Resp()

    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr(llm_client, "call", fake_call)

    out = screen.classify_kinds(conn, cands)
    assert out["ran"] and out["classified"] == 3
    kinds = {c.canonical_id: c.kind for c in cands}
    assert kinds == {"x:user:1": "person", "x:user:2": "org", "x:user:3": "person"}

    # cached on the canonical entity's profile → a fresh rank reads it back
    assert screen.rank_candidates(conn)[0].kind in {"person", "org"}
    # idempotent: re-classify the SAME (now-classified) candidates makes no new LLM call
    fresh = screen.rank_candidates(conn)
    out2 = screen.classify_kinds(conn, fresh)
    assert out2["classified"] == 0 and calls["n"] == 1


def test_classify_degrades_open_when_llm_unavailable(conn, monkeypatch):
    _seed_three(conn)
    cands = screen.rank_candidates(conn)
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: "OPENROUTER_API_KEY not set")

    out = screen.classify_kinds(conn, cands)
    assert out["ran"] is False and out["classified"] == 0
    # nobody classified → everyone stays person-ELIGIBLE (kind None), nothing hidden/demoted
    assert all(c.kind is None and c.is_person for c in cands)


def test_classify_call_exception_is_skip_safe(conn, monkeypatch):
    _seed_three(conn)
    cands = screen.rank_candidates(conn)
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)

    def boom(*a, **k):
        raise RuntimeError("breaker open")

    monkeypatch.setattr(llm_client, "call", boom)
    out = screen.classify_kinds(conn, cands)
    assert out["ran"] is False and all(c.kind is None for c in cands)



def test_classify_keeps_index_alignment_across_batches(conn, monkeypatch):
    """`_parse_verdicts` keys verdicts 1-based into ITS OWN batch and the write does
    `batch[idx - 1]`. A loop that shares one index space across batches, or reorders a batch after
    the call, writes each verdict onto the WRONG person — and there is no reader downstream that
    would notice. 250 candidates = 3 batches, every verdict derived from that line's own name."""
    n = 250
    _seed_named(conn, n)
    state = _patch_llm(monkeypatch)

    cands = screen.rank_candidates(conn)
    assert len(cands) == n
    out = screen.classify_kinds(conn, cands)
    assert out["batches"] == 3 and state["n"] == 3          # ceil(250/100), one call each
    assert out["ran"] is True and out["classified"] == n and out["of"] == n

    # Each verdict landed on the person whose own name asked for it — in memory AND in the cache.
    for c in cands:
        assert c.kind == _KIND_BY_PREFIX[c.name.split("-")[0]], c.name
    for c in screen.rank_candidates(conn):
        assert c.kind == _KIND_BY_PREFIX[c.name.split("-")[0]], c.name


def test_classify_isolates_a_failed_batch(conn, monkeypatch):
    """One bad batch costs its own hundred and not the other 150, and leaves those hundred
    kind=None (person-eligible), so the next screen retries exactly them. Degrade-open per batch is
    strictly stronger than the all-or-nothing it replaced."""
    n = 250
    _seed_named(conn, n)
    _patch_llm(monkeypatch, fail_on_call=2)

    cands = screen.rank_candidates(conn)
    out = screen.classify_kinds(conn, cands)
    assert out["ran"] is True and out["batches"] == 3 and out["failed_batches"] == 1
    assert out["classified"] == 150 and out["of"] == n
    assert "breaker open" in out["reason"]

    # Batch 2 is cands[100:200] — the pending list is the ranked list, sliced in order.
    assert all(c.kind is None for c in cands[100:200])
    assert all(c.kind is not None for c in cands[:100] + cands[200:])
    # …and the skip PERSISTED as unclassified, so a re-screen re-spends on only those 100.
    assert len([c for c in screen.rank_candidates(conn) if c.kind is None]) == 100

# ── assembly: pre-tick + floor (no demotion, no reorder) ───────────────────────

def test_build_screen_preticks_persons_without_demoting_or_reordering(conn, monkeypatch):
    """The label's ONLY consequence is the pre-tick (David, 2026-08-24). This file previously
    asserted the opposite — that a classified org sorted LAST and never counted toward the floor.
    Reversed because the kind is judged from name + bio alone, often a name alone, and moving
    someone out of the default view on that evidence is the closest thing to hiding a real
    person. The ORG here deliberately OUTRANKS the person (higher total_count), so the old
    demotion would have been visible as a reordering."""
    _person(conn, "x:user:p", name="PERSON-1", profile={"bio": "builder"})
    schema.add_signal(conn, "x:user:p", "follow", "x")
    schema.add_signal(conn, "x:user:p", "save", "x", count=4)          # total 5
    _person(conn, "x:user:o", name="ORG-1", profile={"bio": "the company"})
    schema.add_signal(conn, "x:user:o", "follow", "x")
    schema.add_signal(conn, "x:user:o", "list", "x", count=6, extra={"list_names": ["ai"]})  # 7

    _patch_llm(monkeypatch)

    scr = screen.build_screen(conn, floor=1)
    by_id = {c["canonical_id"]: c for c in scr["candidates"]}
    assert by_id["x:user:p"]["pre_ticked"] is True and by_id["x:user:p"]["is_person"]
    # classified org: NOT pre-ticked …
    assert by_id["x:user:o"]["pre_ticked"] is False and by_id["x:user:o"]["is_person"] is False
    assert by_id["x:user:o"]["kind"] == "org"          # still REPORTED, so the host can say so
    # … but keeps its rank position, and still fills the floor.
    ids = [c["canonical_id"] for c in scr["candidates"]]
    assert ids == ["x:user:o", "x:user:p"], "the label must not reorder the list"
    assert by_id["x:user:o"]["shown_by_default"] is True
    assert scr["recommended_count"] == 1 and scr["shown_by_default_count"] == 2


# ── the scholar tier ─────────────────────────────────────────────────────────────

def _cand(cid, *, signals, endorsement=False, distinct=1, count=1):
    return screen.Candidate(canonical_id=cid, name=cid, signals=signals,
                        has_endorsement=endorsement, distinct_signals=distinct,
                        total_count=count)


_FOLLOW = [{"signal_type": "follow", "platform": "x", "count": 1, "extra": None}]
_PAPERS = [{"signal_type": "save", "platform": "openalex", "count": 3, "extra": None}]


def test_a_researcher_is_interleaved_with_follows_not_buried_under_them():
    """OpenAlex has no follow primitive, so a scholar can NEVER earn an endorsement signal and
    sorts below every X follow under `sort_key` alone. That is the ordering being wrong for a
    structural reason. Interleaving by RANK POSITION fixes it without weakening the endorsement
    key — no ratio between "a follow" and "three saved papers" has to be defended."""
    ranked = [_cand("x:1", signals=_FOLLOW, endorsement=True),
              _cand("x:2", signals=_FOLLOW, endorsement=True),
              _cand("openalex:A1", signals=_PAPERS),
              _cand("openalex:A2", signals=_PAPERS)]

    out = [c.canonical_id for c in screen.interleave_tiers(ranked)]

    assert out == ["x:1", "openalex:A1", "x:2", "openalex:A2"]


def test_a_single_platform_user_sees_their_list_untouched():
    ranked = [_cand("x:1", signals=_FOLLOW, endorsement=True), _cand("x:2", signals=_FOLLOW)]
    assert screen.interleave_tiers(ranked) == ranked

    scholars = [_cand("openalex:A1", signals=_PAPERS), _cand("openalex:A2", signals=_PAPERS)]
    assert screen.interleave_tiers(scholars) == scholars


def test_a_person_the_user_also_follows_stays_in_the_main_tier():
    """ALL signals scholar, not ANY. Someone the user follows on X who also wrote a paper they
    saved has a real endorsement, and it should earn them their rank rather than move them into
    the tier for people known only as authors."""
    both = _cand("x:9", signals=_FOLLOW + _PAPERS, endorsement=True, distinct=2)
    assert both.is_scholar is False
    assert _cand("openalex:A1", signals=_PAPERS).is_scholar is True


def test_a_scholar_is_never_pre_ticked_however_corroborated(conn):
    """A pre-tick is OPYT vouching. The user picked the PAPER; inferring a vouch for its author
    from that is the inference OPYT should not make on their behalf."""
    schema.upsert_entity(conn, "openalex:A1", name="F. Arnold")
    schema.set_signal(conn, "openalex:A1", "save", "openalex", count=9)
    schema.set_signal(conn, "openalex:A1", "save", "scholar", count=4)   # 2 distinct → corroborated

    out = screen.build_screen(conn)
    card = next(c for c in out["candidates"] if c["canonical_id"] == "openalex:A1")

    assert card["corroborated"] is True
    assert card["pre_ticked"] is False
    assert card["shown_by_default"] is True          # never hidden — only never vouched for


def test_the_classifier_is_never_called_for_a_scholar(conn, monkeypatch):
    """`kind` decides exactly one thing — the pre-tick — and no scholar is ever pre-ticked, so a
    paid call for them cannot change any outcome."""
    schema.upsert_entity(conn, "openalex:A1", name="F. Arnold")
    schema.set_signal(conn, "openalex:A1", "save", "openalex", count=3)
    called = []
    monkeypatch.setattr(screen, "_classify_prompt", lambda batch: called.append(batch) or "")

    out = screen.classify_kinds(conn, screen.rank_candidates(conn))

    assert called == []
    assert out["classified"] == 0


def test_the_authors_of_saved_papers_are_reflected_as_papers_not_posts():
    """The old `x`/not-`x` binary read every non-X save as a Substack post, so an author of three
    saved PAPERS was reflected back as "saved 3 post(s)" — a claim about content the user never
    saw."""
    assert screen.reflect(_cand("openalex:A1", signals=_PAPERS)) == "you saved 3 of their paper(s)"
    assert screen.reflect(_cand("x:1", signals=[
        {"signal_type": "save", "platform": "x", "count": 2, "extra": None}])) == "bookmarked 2×"


# ── the payload a host can actually read ──────────────────────────────────────

def _followed(conn, eid, *, name, count=1):
    """Someone with a plain follow — a candidate, never pre-ticked (distinct=1)."""
    _person(conn, eid, name=name, profile={"bio": "writes"})
    schema.add_signal(conn, eid, "save", "x", count=count)


def test_the_payload_is_bounded_as_a_total_with_no_exemption(conn, monkeypatch):
    """⚠️ THE TEST THAT REPLACES THE ONE THAT BLESSED THE BUG. Its predecessor asserted "no
    pre-ticked candidate is ever cut" and passed — and that assertion WAS the defect: on a real
    store 179 of 1,070 people are pre-ticked, so exempting them left 181 cards riding and the
    payload came back at 68,476 characters, over the host's limit exactly as before.

    A cap with an unbounded exemption is not a cap. This asserts the total.
    """
    for i in range(30):                       # all pre-ticked: follow + likes = 2 distinct signals
        _person(conn, f"x:user:{i}", name=f"PERSON-{i}", profile={"bio": "writes"})
        schema.add_signal(conn, f"x:user:{i}", "follow", "x")
        schema.add_signal(conn, f"x:user:{i}", "like", "x", count=30 - i)
    _patch_llm(monkeypatch)

    scr = screen.build_screen(conn, floor=15, limit=10)

    assert scr["recommended_count"] == 30         # the vouch is still COUNTED in full …
    assert len(scr["candidates"]) == 10           # … and the payload is still bounded
    assert scr["omitted"]["count"] == 20
    assert sum(1 for c in scr["candidates"] if c["pre_ticked"]) == 10


def test_a_real_sized_store_fits_in_the_reader(conn, monkeypatch):
    """The measurement, not the mechanism. 297 chars/card measured live on 2026-09-14; the two
    payloads that failed were 68KB and 52KB. Any future change to `_card` that pushes a default
    screen back over ~25KB reopens the exact bug."""
    import json

    for i in range(400):
        _person(conn, f"x:user:{i}", name=f"PERSON-{i}", profile={"bio": "a writer of things"})
        schema.add_signal(conn, f"x:user:{i}", "follow", "x")
        schema.add_signal(conn, f"x:user:{i}", "like", "x", count=400 - i)
    _patch_llm(monkeypatch)

    size = len(json.dumps(screen.build_screen(conn)))

    assert size < 25_000, f"a default screen is {size:,} chars — the host could not read 52,000"


def test_an_ordinary_screen_reports_no_omission_at_all(conn, monkeypatch):
    """`omitted: 0` on every ordinary result trains the reader to skip the one field that says the
    list is short."""
    _followed(conn, "x:user:1", name="PERSON-1")
    _patch_llm(monkeypatch)

    assert "omitted" not in screen.build_screen(conn)


def test_the_cap_bounds_what_is_returned_never_what_is_ranked(conn, monkeypatch):
    """Ranks must not move when the cap does, or two renders of one store disagree about who the
    user cares most about. The classify still runs over everybody for the same reason."""
    for i in range(20):
        _followed(conn, f"x:user:{i}", name=f"PERSON-{i}", count=20 - i)
    _patch_llm(monkeypatch)

    narrow = screen.build_screen(conn, limit=8)
    wide = screen.build_screen(conn, limit=100)

    ids = [c["canonical_id"] for c in wide["candidates"]]
    assert [c["canonical_id"] for c in narrow["candidates"]] == ids[:8]
    assert "omitted" not in wide


def test_a_card_carries_the_sentence_and_not_the_rows_behind_it(conn, monkeypatch):
    """`signals`, `identity_links` and `members` WERE the payload, and every one of them is the raw
    form of something the card already states: `reflected` is `signals` as a sentence,
    `distinct_signals` is its length, `canonical_id` is the handle onto the cluster. No reader
    outside `screen` ever read them off a card."""
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    schema.add_signal(conn, "x:user:1", "follow", "x")
    schema.add_signal(conn, "x:user:1", "save", "x", count=12)
    _patch_llm(monkeypatch)

    card = screen.build_screen(conn)["candidates"][0]

    assert card["reflected"] == "you follow · bookmarked 12×"
    assert card["distinct_signals"] == 2
    assert not {"signals", "identity_links", "members"} & set(card)


# ── source= : the bounded form of "show me my Substack people" ──────────────────
#
# THE DEFECT THESE EXIST FOR, read back from a real session 2026-09-14: the user asked to see
# their Substack writers, no surface answered that in bounds, and the host called
# `screen(limit=1100)` — straight past the `omitted` note's "do NOT raise it past ~80". The result
# was 290,754 characters, over the token limit, recovered only by spilling to a file and `jq`-ing
# it. The cap was not the thing that was broken; the missing question was.
def _both_platforms(conn, eid, *, name):
    """Someone the user follows on X AND subscribes to on Substack."""
    _person(conn, eid, name=name, profile={"bio": "writes"})
    schema.add_signal(conn, eid, "follow", "x")
    schema.add_signal(conn, eid, "subscribe", "substack")


def test_source_narrows_who_is_listed(conn, monkeypatch):
    _followed(conn, "x:user:1", name="X-ONLY")
    _person(conn, "substack:sub", name="SUBSTACK-ONLY", profile={"bio": "writes"})
    schema.add_signal(conn, "substack:sub", "subscribe", "substack")
    _patch_llm(monkeypatch)

    out = screen.build_screen(conn, source="substack")

    assert [c["name"] for c in out["candidates"]] == ["SUBSTACK-ONLY"]
    assert out["total_candidates"] == 1 and out["source"] == "substack"


def test_source_is_membership_not_exclusivity(conn, monkeypatch):
    """The corroborated cross-platform people are the whole point of asking. Requiring EVERY
    signal to match would hide exactly the ones a screen exists to surface."""
    _both_platforms(conn, "x:user:1", name="BOTH")
    _patch_llm(monkeypatch)

    assert [c["name"] for c in screen.build_screen(conn, source="substack")["candidates"]] == \
           [c["name"] for c in screen.build_screen(conn, source="x")["candidates"]] == ["BOTH"]


def test_source_narrows_the_list_without_moving_a_rank(conn, monkeypatch):
    """`source=` is a filter, never a re-score. Filtering after ranking is what guarantees it."""
    for i in range(6):
        _followed(conn, f"x:user:{i}", name=f"X-{i}", count=100 - i)
    for i in range(6):
        _person(conn, f"substack:s{i}", name=f"S-{i}", profile={"bio": "writes"})
        schema.add_signal(conn, f"substack:s{i}", "subscribe", "substack", count=50 - i)
    _patch_llm(monkeypatch)

    everyone = [c["canonical_id"] for c in screen.build_screen(conn, limit=80)["candidates"]]
    subs = [c["canonical_id"] for c in screen.build_screen(conn, source="substack")["candidates"]]

    assert subs == [cid for cid in everyone if cid in set(subs)], "filtering reordered the list"


def test_an_unknown_source_says_so_instead_of_returning_nobody(conn, monkeypatch):
    """An empty list IS a guess here: it reads exactly like a connected platform nobody arrives
    through. Naming what the store holds is the repair."""
    _followed(conn, "x:user:1", name="PERSON-1")
    _patch_llm(monkeypatch)

    out = screen.build_screen(conn, source="twitter")

    assert "twitter" in out["error"] and "x" in out["known_platforms"]
    assert "candidates" not in out


def test_an_oversized_limit_is_clamped_and_reported_not_honored(conn, monkeypatch):
    """A documented "do not" the tool cheerfully honors is not a guard — the host raised `limit`
    to 1100 precisely because nothing stopped it."""
    for i in range(120):
        _followed(conn, f"x:user:{i}", name=f"PERSON-{i}", count=200 - i)
    _patch_llm(monkeypatch)

    out = screen.build_screen(conn, limit=1100)

    assert len(out["candidates"]) == screen.SCREEN_LIMIT_MAX
    assert out["limit_clamped"] == {"asked": 1100, "applied": screen.SCREEN_LIMIT_MAX,
                                    "note": out["limit_clamped"]["note"]}
    assert "source=" in out["limit_clamped"]["note"], "say how to ask the bounded question"


def test_an_ordinary_limit_reports_no_clamp(conn, monkeypatch):
    _followed(conn, "x:user:1", name="PERSON-1")
    _patch_llm(monkeypatch)
    assert "limit_clamped" not in screen.build_screen(conn, limit=40)
