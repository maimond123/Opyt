"""The watchlist — the standing queries as a surface a person can see and change.

RULED 2026-08-25 (David's design): standing queries become a user-visible watchlist, reviewed at
the decision point rather than pinned by a flag. Review-at-the-decision-point beats standing pin
state — no flag, no writer exemption, and strictly more power (add and edit, not just keep).

What these lock, all of them silent failures:

  • PULL-ONLY. The list is shown when the user asks and inside the result of a read they themselves
    triggered. A scheduler read records the same diff and surfaces nothing. Standing queries run
    quietly; announcing them unprompted is the recital the frontier surface's etiquette forbids.
  • NO LANE VOCABULARY. The quota is enforcement-internal. Naming it means teaching the entry_mode
    taxonomy to explain a distinction the user cannot act on.
  • `votable=False` ON A USER QUERY IS LOAD-BEARING. `_sync_speed` takes the MIN miss_count over
    VOTABLE claims, and nothing ever verdicts a user-authored query — so a votable user claim sits
    at 0 forever, pins every query it touches to daily, and erases decay through one shared row.
  • A DROP IS GLOBAL. One list of questions, not a copy per region.
  • THE DIFF IS NAMED, NOT COUNTED. "three new questions" leaves the user unable to judge or drop
    any of them.
  • RE-ADDING A DROPPED QUESTION DOES NOT RESTART IT, and says so (added 2026-09-05). The rule it
    reports is correct and must stay — `upsert_queries` never writes `status`, so no machine
    re-emission can undo a human retirement. The failure was reporting `added` for it, alongside a
    watchlist that did not contain it.
  • THE RETIRED LIST IS GATED BEHIND `show='retired'`. It only ever grows, so an always-present
    block gets longer forever for a user who never asks. `unretire` is unusable without it.
"""
from __future__ import annotations

import pytest

from mcp_server import sitting_tools as st
from pipeline.kb import frontier_execute as fe
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema

REGION, OTHER = "sitting:mlx", "sitting:kiss1r"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _offline_first_pull(monkeypatch):
    """Every `add` starts its first pull; stub the executor so this file stays offline, and run
    the spawn SYNCHRONOUSLY — a real daemon thread can outlive the monkeypatching and hit the
    actual network after the stub is gone. The first-pull contract itself is tested at the end
    of this file; the executor's own scoping is tested in test_frontier_execute.py."""
    monkeypatch.setattr(st, "_spawn", lambda target: target())
    monkeypatch.setattr(fe, "run_frontier_execute",
                        lambda conn=None, **kw: {"status": "ok", "candidates_new": 0})


def _q(text: str) -> dict:
    return {"text": text, "rationale": "r", "target_sources": ["arxiv"], "atom_ids": []}


def _call(conn, **kw) -> dict:
    return st._watchlist(conn, **{"sitting_id": None, "query": None, "add": None, "drop": None,
                                  "unretire": None, "show": None, **kw})


def _texts(res) -> list[str]:
    return [w["text"] for w in res["watching"]]


def test_every_user_query_source_has_an_adapter():
    """⚠️ THE INVARIANT THAT WAS VIOLATED FOR MONTHS. `USER_QUERY_SOURCES` is unconditional — every
    name in it runs against every question a user types, with no `target_sources` escape hatch
    (deleted in fe48a993). So a name with no adapter is not an unbuilt to-do, it is a permanent
    `no_adapter` row on every user query forever. `semantic_scholar` and `hackernews` sat there
    until 2026-09-04 and neither had existed in any commit of `frontier_sources`, so two of every
    five pairs were dead on arrival.

    Deliberately NOT asserted against `reader_core.VALID_SOURCES`, which is the reader's proposal
    vocabulary and keeps unbuilt names on purpose so an unserved gap stays visible."""
    from pipeline.kb import frontier_sources
    built = set(frontier_sources.adapters())
    assert set(fq.USER_QUERY_SOURCES) <= built, (
        f"routed at sources with no adapter: {set(fq.USER_QUERY_SOURCES) - built}")


# ── the list ────────────────────────────────────────────────────────────────────
def test_the_list_names_the_speed_and_who_asked(conn):
    """Today the user sees only a query COUNT in frontier status. Speed is what answers "how often
    is this running", and `source` is the one distinction they can act on — only a question they
    typed themselves is exempt from decay."""
    fq.upsert_queries(conn, [_q("continuous batching")], generator=REGION)
    fq.add_user_query(conn, "agentic payment rails")

    rows = {w["text"]: w for w in _call(conn)["watching"]}
    assert rows["continuous batching"]["speed"] == "daily"
    assert rows["continuous batching"]["source"] == "a read of your material"
    assert rows["agentic payment rails"]["source"] == "you"


def test_a_slowed_query_reads_as_slower_not_as_gone(conn):
    """The decay tiers exist so no machine path ever needs to remove a query. A quiet thread slows
    to a cheap monthly floor and stays visible — and a list that hid it would make "why did my
    question disappear" unanswerable."""
    fq.upsert_queries(conn, [_q("quiet thread")], generator=REGION)
    for _ in range(fe.DECAY_TIERS[0][0]):
        fq.apply_verdicts(conn, [{"text": "quiet thread", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"]


def test_no_lane_vocabulary_reaches_the_surface(conn):
    """The quota is enforcement-internal bookkeeping. Telling a person three of their watched
    questions are 'machine lane' requires teaching the whole entry_mode taxonomy to explain a
    distinction that changes nothing they can do."""
    fq.upsert_queries(conn, [dict(_q("verifiable compute"), lane=fq.LANE_MACHINE)],
                      generator=REGION)
    res = _call(conn)
    assert _texts(res) == ["verifiable compute"], "the query itself must still be shown"
    body = repr(res).lower()
    assert not any(w in body for w in ("lane", "machine", "frontier", "quota"))


def test_the_list_scopes_to_a_region_when_one_is_named(conn):
    fq.upsert_queries(conn, [_q("mlx thread")], generator=REGION)
    fq.upsert_queries(conn, [_q("kisspeptin thread")], generator=OTHER)
    assert _texts(_call(conn, query="mlx")) == ["mlx thread"]
    assert sorted(_texts(_call(conn))) == ["kisspeptin thread", "mlx thread"]


def test_asking_about_an_unwatched_topic_builds_nothing(conn):
    """A watchlist request must not quietly buy an embedding and mint a region as a side effect of
    asking what is being watched."""
    res = _call(conn, query="something nobody has read")
    assert res["status"] == "error" and "read that region first" in res["reason"]
    assert conn.execute("SELECT COUNT(*) FROM sittings").fetchone()[0] == 0


# ── add ─────────────────────────────────────────────────────────────────────────
def test_a_user_added_query_is_not_votable(conn):
    """THE HAZARD. `_sync_speed` MINs over VOTABLE claims only, and nothing ever renders a verdict
    on a user-authored query. Votable, the user's claim sits at miss_count 0 forever and pins every
    query it touches to the daily tier — decay dead across the whole set, counters all healthy."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=REGION)
    for _ in range(4):
        fq.apply_verdicts(conn, [{"text": "shared thread", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"]

    _call(conn, add=["shared thread"])                 # the user adopts the same question
    assert conn.execute("SELECT votable FROM frontier_query_generators g "
                        " JOIN frontier_generators fg ON fg.generator = g.generator "
                        " WHERE g.generator = ?", (fq.USER_GENERATOR,)).fetchone()[0] == 0
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"], \
        "the user's claim voted and reset the decay"


def test_an_added_query_says_it_will_not_decay(conn):
    """The pin reborn as the obvious semantics of an add button — and said out loud, because a
    question that silently behaved differently from its neighbours is worse than no exemption."""
    res = _call(conn, add=["agentic payment rails"])
    assert res["added"] == ["agentic payment rails"]
    assert res["added_note"]
    assert "agentic payment rails" in _texts(res)


# ── drop ────────────────────────────────────────────────────────────────────────
def test_a_drop_retires_the_question_everywhere(conn):
    """One list of questions, not a copy per region — so a question two regions both watch is
    retired for both. Stated in the result, because the user asked from inside one region and would
    otherwise have no way to know."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=REGION)
    fq.upsert_queries(conn, [_q("shared thread")], generator=OTHER)

    res = _call(conn, query="mlx", drop=["shared thread"])
    assert res["dropped"] == ["shared thread"]
    assert res["dropped_note"]
    assert _texts(_call(conn)) == []


def test_a_drop_that_matched_nothing_is_reported(conn):
    """SHOW DECIDED, DON'T HIDE. A silent no-op reads as success, and the user walks away believing
    they stopped watching something they did not."""
    res = _call(conn, drop=["never existed"])
    assert res["not_found"] == ["never existed"]
    assert "dropped" not in res


def test_nothing_but_a_human_retires_a_query(conn):
    """No omission-retirement machinery of any kind — and none to exempt user queries FROM. A query
    the reader stops re-emitting only slows; the retired list is therefore always a person's doing,
    which is what makes it a meaningful third bucket in the diff."""
    fq.upsert_queries(conn, [_q("dropped every time")], generator=REGION)
    for _ in range(20):
        fq.apply_verdicts(conn, [{"text": "dropped every time", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert fq.retired_texts(conn, generator=REGION) == []
    assert [w["speed"] for w in _call(conn)["watching"]] == ["monthly"]


# ── re-add, unretire, and seeing what is retired ────────────────────────────────
def test_re_adding_a_dropped_question_says_so_instead_of_claiming_it_was_added(conn):
    """THE DEFECT, in one test (measured live 2026-09-05). `upsert_queries` never writes `status`,
    which is correct — a machine re-emission must not resurrect a human retirement. But the USER
    re-typing the question was treated as a machine re-emission: the response reported `added` and
    returned an EMPTY watchlist in the same dict."""
    _call(conn, add=["agentic payment rails"])
    _call(conn, drop=["agentic payment rails"])

    res = _call(conn, add=["agentic payment rails"])
    assert res["still_retired"] == ["agentic payment rails"]
    assert "added" not in res                     # the negative is the whole point
    assert _texts(res) == []                      # and it agrees with the list beside it
    assert fq.retired_texts(conn) == ["agentic payment rails"]


def test_unretire_puts_a_dropped_question_back(conn):
    """The inverse of `drop`, and the reason it now lives on this surface rather than on a CLI
    flag: the destructive direction was reachable and the undo was not."""
    _call(conn, add=["verification games"])
    _call(conn, drop=["verification games"])

    res = _call(conn, unretire=["verification games"])
    assert res["unretired"] == ["verification games"]
    assert _texts(res) == ["verification games"]
    assert fq.retired_texts(conn) == []


def test_an_unretire_that_matched_nothing_is_reported(conn):
    """Same rule as a drop that matched nothing: the match is on exact text, so a paraphrase
    silently restores nothing and the user must be told."""
    res = _call(conn, unretire=["never existed"])
    assert res["not_found"] == ["never existed"]
    assert "unretired" not in res


def test_retired_questions_are_hidden_until_asked_for(conn):
    """The default response must not grow. The retired set only accumulates, so an always-present
    block would get longer forever for a user who never asks."""
    _call(conn, add=["agentic payment rails"])
    _call(conn, drop=["agentic payment rails"])

    assert "retired" not in _call(conn)
    assert _call(conn, show="retired")["retired"] == ["agentic payment rails"]
    assert _call(conn, show="retired")["retired_note"]


def test_show_takes_only_retired(conn):
    """A value this does not understand is an error, not a silently ignored argument — a caller
    that guesses `show='all'` would otherwise be told nothing and shown nothing."""
    res = _call(conn, show="all")
    assert res["status"] == "error"
    assert "retired" in res["reason"]


def test_a_region_whose_questions_are_all_retired_can_still_be_asked_about(conn):
    """Scoping tests whether the REGION exists, not whether anything active survives in it. A
    region you have emptied by dropping is exactly the one you ask `show='retired'` about, and
    testing only the active list would refuse the request that needs answering most."""
    fq.upsert_queries(conn, [_q("mlx kernel fusion")], generator=REGION)
    _call(conn, drop=["mlx kernel fusion"])

    res = _call(conn, query="mlx", show="retired")
    assert res["status"] == "ok"
    assert res["retired"] == ["mlx kernel fusion"]


def test_the_retired_list_scopes_to_a_region_when_one_is_named(conn):
    """A drop is global, but "what did I drop" asked from inside a region is about that region."""
    fq.upsert_queries(conn, [_q("mlx kernel fusion")], generator=REGION)
    fq.upsert_queries(conn, [_q("kiss1r rollout")], generator=OTHER)
    _call(conn, drop=["mlx kernel fusion"])
    _call(conn, drop=["kiss1r rollout"])

    assert _call(conn, query="mlx", show="retired")["retired"] == ["mlx kernel fusion"]
    assert set(_call(conn, show="retired")["retired"]) == {"mlx kernel fusion", "kiss1r rollout"}


# ── the first pull ──────────────────────────────────────────────────────────────
#
# ⚠️ BLOCK ON DECISIONS, NEVER ON MACHINE WORK (RULED 2026-09-12, David). The first cut ran
# the pull in the foreground; a live user added ten topics in one call and stared at a spinner
# for five minutes. Now the add returns immediately and the pull runs behind the conversation.
# These lock the contract's four properties: it starts, it is scoped, it never blocks or
# breaks the add, and the response promises nothing the pull has not yet done.

def test_an_add_starts_its_first_pull_in_the_background_and_promises_nothing(conn, monkeypatch):
    seen = {}

    def fake(conn=None, **kw):
        seen["conn"] = conn
        seen.update(kw)
        return {"status": "ok", "candidates_new": 7}
    monkeypatch.setattr(fe, "run_frontier_execute", fake)

    out = _call(conn, add=["sim-to-real failure recovery"])

    assert seen["query_ids"] == {fq.query_id_for(fq.normalize("sim-to-real failure recovery"))}
    # Its OWN connection: the caller's is mid-request and SQLite is not thread-shared.
    assert seen["conn"] is None
    assert out["first_pull"]["status"] == "running"
    # No counts and no findings: the response is written before the pull finishes, so a claim
    # about what landed would be a guess. The note says where results WILL surface instead.
    assert "found" not in out["first_pull"]
    assert "frontier" in out["first_pull"]["note"]
    assert "background" in out["first_pull"]["note"]


def test_the_first_pull_is_scoped_to_what_was_just_added(conn, monkeypatch):
    """An add on an established store must not piggyback every other due standing query — the
    scheduled rail owns those."""
    fq.upsert_queries(conn, [_q("an older standing question")], generator="bookmark-reader")
    seen = {}
    monkeypatch.setattr(fe, "run_frontier_execute",
                        lambda conn=None, **kw: seen.update(kw) or {"status": "ok",
                                                                    "candidates_new": 0})

    _call(conn, add=["a brand new question"])

    assert fq.query_id_for(fq.normalize("an older standing question")) not in seen["query_ids"]


def test_the_adds_are_committed_before_the_pull_thread_opens_its_own_connection(conn,
                                                                                monkeypatch):
    """The pull runs on a NEW connection, which sees only what is on disk. If the add were
    still an open transaction on the caller's connection, the pull would find no such query
    and silently pull nothing — correct-looking, empty forever."""
    seen = {}

    def fake(conn=None, **kw):
        from pipeline.kb import schema
        own = schema.connect()
        try:
            seen["visible"] = own.execute(
                "SELECT COUNT(*) FROM frontier_queries WHERE query_id IN (%s)"
                % ",".join("?" * len(kw["query_ids"])), tuple(kw["query_ids"])).fetchone()[0]
        finally:
            own.close()
        return {"status": "ok", "candidates_new": 0}
    monkeypatch.setattr(fe, "run_frontier_execute", fake)

    _call(conn, add=["must be on disk"])

    assert seen["visible"] == 1


def test_a_failed_spawn_never_breaks_the_add(conn, monkeypatch):
    """The add already succeeded when the spawn happens; a thread that cannot start must not
    dress it up as a failure. The scheduled rail pulls the same pairs later regardless."""
    def boom(target):
        raise RuntimeError("no threads today")
    monkeypatch.setattr(st, "_spawn", boom)

    out = _call(conn, add=["still gets added"])

    assert out["status"] == "ok"
    assert "still gets added" in out["added"]
    assert "first_pull" not in out


def test_a_pull_that_dies_in_the_thread_is_swallowed(conn, monkeypatch):
    """The thread body is the last fail-safe layer: whatever escapes the executor dies there,
    never in the user's response — and the spawn already happened, so the note stands."""
    def boom(conn=None, **kw):
        raise RuntimeError("network down mid-pull")
    monkeypatch.setattr(fe, "run_frontier_execute", boom)

    out = _call(conn, add=["still gets added"])

    assert out["status"] == "ok" and out["first_pull"]["status"] == "running"


def test_a_drop_runs_no_pull(conn, monkeypatch):
    _call(conn, add=["to be dropped"])
    monkeypatch.setattr(fe, "run_frontier_execute",
                        lambda conn=None, **kw: pytest.fail("a drop must not pull"))

    _call(conn, drop=["to be dropped"])


# ── the ending, owned ───────────────────────────────────────────────────────────

def test_an_add_into_an_empty_store_hands_the_host_next_steps(conn):
    """⚠️ THE 2026-09-12 TRANSCRIPT, last turn: six watches in, store empty, and the host
    closed with no next steps at all — a success report over a product state of "nothing to
    read". The response now carries the one fact the host cannot see (atoms == 0) and the
    instruction to end with the content routes."""
    out = _call(conn, add=["a watch into the void"])

    assert "next steps" in out["store_note"]


def test_a_store_with_material_gets_no_store_note(conn):
    import json as _json
    conn.execute(
        "INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode, what_kind, "
        "description, payload) VALUES ('a:1','substack','x:a','2026-09-01','user-saved',"
        "'post','p', ?)", (_json.dumps({}),))
    conn.commit()

    out = _call(conn, add=["a watch with company"])

    assert "store_note" not in out
