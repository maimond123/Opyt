"""Stage-4 Oracle SCREEN — ranking, the kind classifier (degrade-open), and the recommended/
see-all partition. Pure over entities + curation_signals; the LLM is monkeypatched (no network,
no key), so the classify LOGIC (batching, per-batch degrade-open, index alignment,
cache, idempotency) is proven offline."""
from __future__ import annotations

import json
import re

import pytest

from pipeline.kb import resolve, schema, screen


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
