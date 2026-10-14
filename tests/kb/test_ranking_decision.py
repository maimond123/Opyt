"""`ranking.rank`, the ranking decision, on plain rows: no store, no clock."""
from __future__ import annotations

from pipeline.oracles import ranking



def _row(cid, eid, signal_type, platform, count=1, name=None):
    return {"canonical_id": cid, "entity_id": eid, "signal_type": signal_type,
            "platform": platform, "count": count, "extra": None, "name": name,
            "identity_links": None, "profile": None}


def test_rows_group_into_one_candidate_per_canonical_id():
    out = ranking.rank([_row("a", "x:1", "like", "x"), _row("a", "sub:1", "save", "substack"),
                       _row("b", "x:2", "like", "x")], set())
    assert [(c.canonical_id, c.distinct_signals, c.members) for c in out] == [
        ("a", 2, ["x:1", "sub:1"]), ("b", 1, ["x:2"])]


def test_an_endorsement_outranks_more_content_signals():
    out = ranking.rank([_row("a", "x:1", "like", "x"), _row("a", "x:1", "save", "x"),
                       _row("b", "x:2", "follow", "x")], set())
    assert [c.canonical_id for c in out] == ["b", "a"]


def test_a_cluster_is_retired_only_when_every_maintained_member_is():
    follow = (ranking.MAINTAINED.signal, ranking.MAINTAINED.platform)
    rows = [_row("a", "x:1", *follow), _row("b", "x:2", *follow), _row("b", "x:3", *follow)]
    out = {c.canonical_id: c.retired for c in ranking.rank(rows, {"x:1", "x:2"})}
    assert out == {"a": True, "b": False}


def test_nothing_retires_a_cluster_without_a_maintained_signal():
    out = ranking.rank([_row("a", "x:1", "like", "x")], {"x:1"})
    assert [c.retired for c in out] == [False]


def test_members_list_the_whole_cluster_not_only_signal_bearers():
    """An X account merged into a Substack-only person carries no signal, and scouting must see it."""
    out = ranking.rank([_row("a", "substack:a", "subscribe", "substack")], set(),
                       clusters={"a": ["substack:a", "x:user:7"]})
    assert out[0].members == ["substack:a", "x:user:7"]
    assert out[0].distinct_signals == 1
