"""GET /v1/kb/{owner}/feed — the arrivals feed a plain HTTP client polls.

Same posture as the rest of this directory: a real export, the real app, a real redeemed reader
token. The properties pinned here are the ones a poller depends on: the body is always ascending,
the cursor walk covers everything exactly once, an exhausted poll echoes its cursor back, and the
auth boundary is the same one the other reads enforce.
"""
from __future__ import annotations

from pipeline.kb.schema import HUMAN_ATTESTED


def _get(svc, **params):
    return svc.client.get(f"/v1/kb/{svc.owner}/feed", params=params, headers=svc.reader_hdr)


def test_seed_page_is_ascending_and_carries_the_card_fields(svc):
    r = _get(svc)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kb"] == svc.owner
    assert body["items"], "the seeded corpus has human-attested atoms; the feed must see them"
    assert body["more"] is False
    for item in body["items"]:
        for key in ("atom_id", "description", "source_url", "who_id", "who",
                    "when_ts", "seen_at", "source_type", "what_kind", "entry_mode"):
            assert key in item
        assert item["entry_mode"] in HUMAN_ATTESTED
    keys = [(i["seen_at"], i["atom_id"]) for i in body["items"]]
    assert keys == sorted(keys), "the body is always ascending, seed page included"
    assert body["next"] == {"since": body["items"][-1]["seen_at"],
                            "since_id": body["items"][-1]["atom_id"]}


def test_cursor_walk_covers_everything_exactly_once(svc):
    everything = _get(svc, since="0000-01-01", limit=200).json()["items"]
    assert everything, "walking from the epoch must see the whole human-attested corpus"

    walked, cursor, hops = [], {"since": "0000-01-01"}, 0
    while True:
        page = _get(svc, limit=2, **cursor).json()
        walked.extend(page["items"])
        hops += 1
        assert hops < 50, "the walk must terminate"
        if not page["more"]:
            # the tail may be a full page (`more` is a probe, not a count); one more poll from
            # its cursor must come back empty before the walk is really done
            if page["items"]:
                cursor = {"since": page["next"]["since"], "since_id": page["next"]["since_id"]}
                continue
            break
        cursor = {"since": page["next"]["since"], "since_id": page["next"]["since_id"]}
    assert [i["atom_id"] for i in walked] == [i["atom_id"] for i in everything]


def test_exhausted_poll_echoes_the_cursor_back(svc):
    seed = _get(svc, limit=200).json()
    nxt = seed["next"]
    empty = _get(svc, since=nxt["since"], since_id=nxt["since_id"]).json()
    assert empty["items"] == []
    assert empty["more"] is False
    assert empty["next"] == {"since": nxt["since"], "since_id": nxt["since_id"]}, \
        "a poller that finds nothing must not lose its place"


def test_date_filter_is_publication_not_arrival(svc):
    """Everything in the corpus ARRIVED just now (same bulk insert), so an arrival window
    keeps it all — the publication filter is what separates backfill from news."""
    everything = _get(svc, since="0000-01-01", limit=200).json()["items"]
    recent = _get(svc, since="0000-01-01", limit=200, date_from="2025").json()["items"]
    assert recent and len(recent) < len(everything)
    assert all(i["when_ts"] >= "2025-01-01" for i in recent)
    assert any(not i["when_ts"] for i in everything), "the corpus seeds an undated atom"
    assert all(i["when_ts"] for i in recent), "an undated atom satisfies no date claim"


def test_date_window_that_matches_nothing_keeps_the_cursor(svc):
    empty = _get(svc, since="0000-01-01", date_from="1901", date_to="1902").json()
    assert empty["items"] == [] and empty["more"] is False
    assert empty["next"] == {"since": "0000-01-01", "since_id": None}


def test_malformed_date_is_a_400_not_a_dropped_filter(svc):
    r = _get(svc, date_from="5/11/2026")
    assert r.status_code == 400


def test_entry_mode_param_narrows(svc):
    only_saved = _get(svc, entry_mode="user-saved").json()["items"]
    assert only_saved
    assert {i["entry_mode"] for i in only_saved} == {"user-saved"}


def test_feed_is_reader_only(svc):
    assert svc.client.get(f"/v1/kb/{svc.owner}/feed").status_code == 401
    assert svc.client.get(f"/v1/kb/{svc.owner}/feed",
                          headers=svc.owner_hdr).status_code == 403


def test_feed_reads_are_counted_by_tool(svc):
    _get(svc)
    by_tool = svc.client.get("/v1/stats").json()["reads_by_tool"]
    assert by_tool.get("feed", 0) >= 1


def test_absurd_limit_is_clamped_not_fatal(svc):
    r = _get(svc, limit=10_000_000)
    assert r.status_code == 200
    assert len(r.json()["items"]) <= 200
