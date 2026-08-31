"""W0 substrate capture — engagement observations from an Oracle's X footprint pull.

Two layers, mirroring test_ingest_x_footprint:
  • PURE extraction tests hit `extract_engagements` directly (no DB / network). They encode the
    capture rules: reply targets from `inReplyToUserId` (numeric ids, never rendered handles),
    self-replies are thread continuations not engagements, quote targets from the `quoted_tweet`
    OBJECT (never the rendered markdown), mentions from ALL thread tweets, unknown author → no rows.
  • DB tests prove `record_engagements` idempotency (re-run writes zero new rows) and the
    fail-safe seam: a capture failure never blocks the atom write.
"""
from __future__ import annotations

import pytest

from pipeline.kb import ingest_x_footprint as fp
from pipeline.kb import schema


_AUTHOR = "99"          # the timeline owner's numeric id (matches the footprint fixtures)


def _tw(tid, *, text="", reply_to_uid=None, rt=False, quoted=None, mentions=None,
        author_id=_AUTHOR, created="Mon Jan 06 10:00:00 +0000 2026"):
    """A raw-twitterapi-shaped tweet carrying the three engagement surfaces."""
    t = {"id": tid, "text": text, "conversationId": tid,
         "author": {"id": author_id, "userName": "carol"},
         "createdAt": created,
         "isRetweet": rt, "isReply": reply_to_uid is not None, "inReplyToUserId": reply_to_uid}
    if quoted is not None:
        t["quoted_tweet"] = quoted
    if mentions is not None:
        t["entities"] = {"user_mentions": mentions}
    return t


def _keys(rows):
    return {(r["kind"], r["target_id"], r["src_ref"]) for r in rows}


# ── PURE: extraction ──────────────────────────────────────────────────────────────

def test_reply_target_extracted_and_self_reply_excluded():
    raw = [_tw("1", reply_to_uid="555"),            # reply to another → captured
           _tw("2", reply_to_uid=_AUTHOR)]          # self-reply = continuation → NOT an engagement
    rows = fp.extract_engagements(raw)
    assert _keys(rows) == {("reply", "x:user:555", "1")}
    assert rows[0]["observer_id"] == f"x:user:{_AUTHOR}"
    assert rows[0]["observed_at"] == "2026-01-06"    # the TWEET's date, not wall clock


def test_quote_target_from_object_not_markdown():
    # The quoted author's numeric id comes off the quoted_tweet OBJECT. Text mentioning a
    # handle (what the rendered markdown carries) must contribute nothing.
    raw = [_tw("1", text="see https://x.com/somebody/status/9",
               quoted={"id": "9", "author": {"id": "777", "userName": "somebody"}})]
    rows = fp.extract_engagements(raw)
    assert _keys(rows) == {("quote", "x:user:777", "1")}


def test_quote_falls_back_to_handle_when_id_absent():
    raw = [_tw("1", quoted={"id": "9", "author": {"userName": "handleonly"}})]
    assert _keys(fp.extract_engagements(raw)) == {("quote", "x:@handleonly", "1")}


def test_self_quote_excluded():
    raw = [_tw("1", quoted={"id": "9", "author": {"id": _AUTHOR, "userName": "carol"}})]
    assert fp.extract_engagements(raw) == []


def test_mentions_from_all_thread_tweets_and_self_mention_excluded():
    raw = [_tw("1", mentions=[{"screen_name": "alice", "id_str": "111"}]),
           _tw("2", reply_to_uid=_AUTHOR,                     # a self-reply still yields ITS mentions
               mentions=[{"screen_name": "bob"},              # no id → handle target, resolved later
                         {"screen_name": "carol", "id_str": _AUTHOR}])]   # self → excluded
    rows = fp.extract_engagements(raw)
    assert ("mention", "x:user:111", "1") in _keys(rows)
    assert ("mention", "x:@bob", "2") in _keys(rows)
    assert not any(r["target_id"] == f"x:user:{_AUTHOR}" for r in rows)


def test_self_mention_by_handle_excluded():
    raw = [_tw("1", mentions=[{"screen_name": "Carol"}])]     # own handle, case-insensitive
    assert fp.extract_engagements(raw) == []


def test_unknown_author_writes_nothing():
    raw = [_tw("1", reply_to_uid="555", author_id=None)]
    raw[0]["author"] = {}                                     # no numeric author id anywhere
    assert fp.extract_engagements(raw) == []


def test_retweets_and_duplicate_pages_contribute_nothing_twice():
    t = _tw("1", reply_to_uid="555")
    raw = [t, dict(t), _tw("2", rt=True, reply_to_uid="666")]
    assert _keys(fp.extract_engagements(raw)) == {("reply", "x:user:555", "1")}


# ── DB: record_engagements idempotency ─────────────────────────────────────────────

@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def test_rerun_writes_zero_new_rows(conn):
    rows = fp.extract_engagements([_tw("1", reply_to_uid="555"),
                                   _tw("2", quoted={"id": "9", "author": {"id": "777"}})])
    assert schema.record_engagements(conn, rows) == 2
    assert schema.record_engagements(conn, rows) == 0          # idempotent natural key
    assert conn.execute("SELECT COUNT(*) FROM engagements").fetchone()[0] == 2


def test_first_observation_date_wins(conn):
    row = {"observer_id": "x:user:99", "kind": "reply", "target_id": "x:user:5",
           "src_ref": "1", "observed_at": "2026-01-06"}
    schema.record_engagements(conn, [row])
    schema.record_engagements(conn, [dict(row, observed_at="2026-02-01")])
    got = conn.execute("SELECT observed_at FROM engagements").fetchone()[0]
    assert got == "2026-01-06"


# ── Fail-safe: capture failure never blocks the atom write ─────────────────────────

def test_atom_still_lands_when_capture_raises(conn, fake_embedder, monkeypatch):
    from tests.kb.test_ingest_x_footprint import _patch_fetch, _raw
    _patch_fetch(monkeypatch, [_raw("10", text="x" * 220)])
    monkeypatch.setattr(fp, "extract_engagements",
                        lambda raw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["added"] == 1                                   # the atom write was unaffected
    assert out["engagements"] == 0


def test_footprint_pull_captures_engagements(conn, fake_embedder, monkeypatch):
    """The wiring: a real pull records the reply/quote/mention observations of ALL fetched
    tweets — including the reply-to-other the curation filter then drops from the atom path."""
    from tests.kb.test_ingest_x_footprint import _patch_fetch, _raw
    dropped_reply = _raw("20", text="@other you are wrong", reply_to_uid="555")
    quote = _raw("21", text="x" * 220,
                 quoted={"id": "9", "author": {"id": "777", "userName": "somebody"}})
    _patch_fetch(monkeypatch, [dropped_reply, quote])
    out = fp.sync_x_footprint(conn, fake_embedder, handle="carol")
    assert out["engagements"] == 2
    got = {(r["kind"], r["target_id"]) for r in
           conn.execute("SELECT kind, target_id FROM engagements")}
    assert got == {("reply", "x:user:555"), ("quote", "x:user:777")}
