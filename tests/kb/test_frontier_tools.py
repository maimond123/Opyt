"""Frontier stage 4 — the MCP delivery layer over `frontier_surface`.

Replaces `tests/artifacts/test_frontier_tools.py`, which covered the v1 shape (a taste block, a
save instruction, a vault-JSON seen-set) that this rewrite removed.

What delivery owes on top of ranking:
  • IT PAGINATES, IT DOES NOT TRUNCATE. `remaining` is reported, so the host can ask for the rest.
    v1's discipline, and the reason it is here: the host cannot rescue what it is never told about.
  • A DISMISSAL LANDS BEFORE THE READ, so an item dismissed in a call comes back IN THAT CALL
    labelled — rather than vanishing between the request and the answer.
  • IT RECORDS WHAT IT SHOWED, once per delivered row, and only for rows it actually delivered.
  • AN EMPTY STORE IS A NOTE, NOT AN ERROR.
"""
from __future__ import annotations

import json

import pytest

from mcp_server import frontier_tools as ft
from pipeline.kb import frontier_surface as fs
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _cand(conn, cid, *, source="arxiv", published="2026-08-11", summary="s" * 1200, payload=None):
    conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES (?,?,?,?,?,?,?,'new',?,?)""",
        (cid, source, f"title {cid}", f"https://example/{cid}", published, summary,
         json.dumps(payload or {}), "2026-08-11", "2026-08-11"))
    conn.commit()
    return cid


def test_deliver_ranks_records_shown_and_reports_the_remainder(conn):
    for i in range(7):
        _cand(conn, f"arxiv:{i}")

    out = ft.deliver(limit=3, conn=conn)

    assert out["status"] == "ok"
    assert out["showing"] == 3 and out["total"] == 7 and out["remaining"] == 4
    assert len(out["candidates"]) == 3
    shown = {r["candidate_id"] for r in conn.execute(
        "SELECT candidate_id FROM frontier_candidate_events WHERE event='shown'")}
    assert shown == {c["candidate_id"] for c in out["candidates"]}, "only delivered rows are shown"


def test_calling_again_advances_because_being_shown_demotes(conn):
    """READING IS A WRITE HERE, and that is the feature. `deliver` records `shown`, the attention
    term demotes on it, so the next call surfaces the NEXT batch instead of re-pitching the same
    head. There is no cursor and no offset — the demotion IS the pagination."""
    for i in range(10):
        _cand(conn, f"arxiv:{i:02d}")

    first = [c["candidate_id"] for c in ft.deliver(limit=4, conn=conn)["candidates"]]
    second = [c["candidate_id"] for c in ft.deliver(limit=4, conn=conn)["candidates"]]

    assert not set(first) & set(second), "a fresh candidate outranks one already shown"


def test_repeated_calls_starve_nothing(conn):
    """The property that matters more than page alignment. An unshown candidate carries ZERO
    attention penalty, so it floats above everything already seen and cannot be stranded below
    the cut forever. 'You cannot notice a thing you were never shown' is the failure this
    forecloses."""
    for i in range(10):
        _cand(conn, f"arxiv:{i:02d}", published=f"2026-0{1 + i % 8}-01")

    seen = set()
    for _ in range(3):                      # ceil(10/4) calls is enough to sweep the queue
        seen |= {c["candidate_id"] for c in ft.deliver(limit=4, conn=conn)["candidates"]}

    assert len(seen) == 10, f"starved: {10 - len(seen)} candidate(s) never surfaced"


def test_dismiss_is_recorded_and_the_row_comes_back_in_the_same_response(conn):
    """The ordering inside `deliver` is the point: dismissals are written first, so the response
    can SHOW the decision instead of the item silently disappearing."""
    _cand(conn, "arxiv:keep")
    _cand(conn, "arxiv:stop")

    out = ft.deliver(dismiss=["arxiv:stop"], conn=conn)

    assert out["dismissed"] == 1
    by_id = {c["candidate_id"]: c for c in out["candidates"]}
    assert "arxiv:stop" in by_id, "a dismissed item must not vanish from the call that dismissed it"
    assert by_id["arxiv:stop"]["state"] == "dismissed"
    assert [c["candidate_id"] for c in out["candidates"]][-1] == "arxiv:stop"


def test_include_dismissed_false_reports_exactly_what_it_hid(conn):
    _cand(conn, "arxiv:a")
    _cand(conn, "arxiv:b")
    fs.record_dismissed(conn, ["arxiv:b"])

    out = ft.deliver(include_dismissed=False, conn=conn)
    assert [c["candidate_id"] for c in out["candidates"]] == ["arxiv:a"]
    assert out["hidden_by_include_dismissed"] == 1, "even the opt-out is not silent"

    assert "hidden_by_include_dismissed" not in ft.deliver(conn=conn)


def test_an_empty_store_is_a_note_not_an_error(conn):
    out = ft.deliver(conn=conn)
    assert out["status"] == "ok" and out["candidates"] == []
    assert "NO CANDIDATES STAGED" in out["note"]
    assert "not an error" in out["note"]


def test_a_card_carries_state_and_reasons_but_no_stored_score(conn):
    _cand(conn, "repo:x", source="github", summary="one-liner", payload={"stars": 4200})
    card = ft.deliver(conn=conn)["candidates"][0]

    assert card["state"] == "new" and card["shown_before"] == 0
    assert "4200 stars" in card["why"]
    assert isinstance(card["score"], float)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(frontier_candidates)")}
    assert "score" not in cols, "the ranking is recomputed per call, never persisted"


def test_limit_zero_shows_nothing_and_records_nothing(conn):
    _cand(conn, "arxiv:a")
    out = ft.deliver(limit=0, conn=conn)
    assert out["showing"] == 0 and out["remaining"] == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidate_events").fetchone()[0] == 0


def test_notice_is_none_when_the_store_is_unreachable(monkeypatch):
    """It rides on `search`. A broken store costs the notice, never the search."""
    from pipeline.kb import schema as sch
    monkeypatch.setattr(sch, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert ft.notice() is None


def test_notice_reports_the_unshown_count(kb_home):
    c = schema.connect()
    _cand(c, "arxiv:a")
    _cand(c, "arxiv:b")
    c.close()
    assert ft.notice()["unshown"] == 2
    ft.deliver(limit=1)
    assert ft.notice()["unshown"] == 1


# ── Registration ────────────────────────────────────────────────────────────────
class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_registers_exactly_the_pull_tool():
    """ONE tool, and the assertion is `==` so a re-added writer fails here.

    ⚠️ THIS TEST'S PREMISE INVERTED ON 2026-08-13. It used to assert `{"frontier", "save_repo"}`
    and pinned `save_repo` so "the deletion step cannot take it by accident" — the deletion was
    deliberate in the end. `save_repo` was cut after measurement: it had never once run, because
    `{vault}/repos/` never existed on disk and its writer `mkdir`s that directory unconditionally
    on entry. So this module now registers a pull tool and nothing else, which is the frontier's
    whole contract — it INFORMS, and admission is stage 3's, autonomously.
    """
    mcp = _FakeMCP()
    ft.register_frontier_tools(mcp)
    assert set(mcp.tools) == {"frontier"}


def test_the_tool_docstring_does_not_promise_an_admission_path(kb_home):
    """A docstring implying the host can 'keep' something would have it tell the user an item was
    added when this tool wrote nothing.

    ⚠️ THIS TEST'S PREMISE CHANGED ON 2026-08-13 and the change is the point. It used to assert
    `"NOT BUILT YET" in doc`, which was true and correct while stage 3 did not exist. Building
    stage 3 made that string a LIE the tool told the host on every call — the docstring instructed
    the model to report that nothing had been added, at exactly the moment things started being
    added. The durable invariant was never "stage 3 is unbuilt"; it is "the HOST has no admission
    path here, so it must not claim one." That is what is pinned now, and it survives stage 3
    existing. See docs/plans/2026-08-12-frontier-stage3-admit.md (AS BUILT)."""
    mcp = _FakeMCP()
    ft.register_frontier_tools(mcp)
    doc = mcp.tools["frontier"].__doc__
    assert "NOT BUILT YET" not in doc                     # the stale claim must not come back
    assert "AUTONOMOUS" in doc                            # ...replaced by what is actually true
    assert "never as \"not good enough\"" in doc          # `rejected` is mechanical, never quality
    assert "Nothing is ever filtered out" in doc
