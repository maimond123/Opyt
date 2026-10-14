"""`readers.match` and the roster — one test per row of the table in the plan
(docs/plans/2026-10-04-restructure-sharing.md, "Decisions")."""
from __future__ import annotations

from pipeline.sharing import readers


def _token(label, sha, role="reader", created="2026-10-04 12:00:00"):
    return {"label": label, "token_sha256": sha, "role": role, "created_at": created}


LEO = _token("Leo", "a1b2c3d4" + "0" * 56)
MAYA = _token("Maya", "ffee0011" + "1" * 56)
NAMELESS = _token(None, "0123abcd" + "2" * 56)
OWNER = _token("David", "9" * 64, role="owner")
HELD = [LEO, MAYA, NAMELESS]


def test_the_owner_token_is_not_a_reader():
    assert readers.readers_of([OWNER, *HELD]) == HELD


def test_a_label_matches_in_any_case():
    assert readers.match(HELD, "  lEo ") == [LEO]


def test_eight_hex_characters_match_a_hash_prefix():
    assert readers.match(HELD, "0123abcd") == [NAMELESS]


def test_a_shorter_string_matches_no_hash():
    assert readers.match(HELD, "0123abc") == []


def test_an_empty_name_never_matches_an_unlabelled_reader():
    assert readers.match(HELD, "") == []
    assert readers.match(HELD, "   ") == []


def test_two_readers_with_one_label_both_come_back():
    twin = _token("leo", "b" * 64)
    assert readers.match([*HELD, twin], "Leo") == [LEO, twin]


def test_the_roster_names_each_reader_by_label_and_short_id():
    assert readers.roster([LEO, NAMELESS]) == [
        {"label": "Leo", "id": "a1b2c3d40000", "since": "2026-10-04 12:00:00"},
        {"label": None, "id": "0123abcd2222", "since": "2026-10-04 12:00:00"},
    ]
    assert readers.name(LEO) == "Leo" and readers.name(NAMELESS) == "0123abcd2222"


# ── an agent's key: in the roster with the readers, marked, and worded as what it is ──────

AGENT = _token("Instinct", "c0ffee00" + "3" * 56, role="agent")


def test_an_agent_is_in_the_roster_with_the_readers_and_the_owner_is_not():
    assert readers.readers_of([OWNER, LEO, AGENT]) == [LEO, AGENT]


def test_the_roster_marks_an_agent_and_leaves_a_reader_as_it_was():
    assert readers.roster([LEO, AGENT]) == [
        {"label": "Leo", "id": "a1b2c3d40000", "since": "2026-10-04 12:00:00"},
        {"label": "Instinct", "id": "c0ffee003333", "since": "2026-10-04 12:00:00",
         "agent": True},
    ]


def test_an_agent_is_picked_by_its_name_or_its_id_like_a_reader():
    held = [LEO, AGENT]
    assert readers.pick(held, "instinct") == readers.Picked(AGENT)
    assert readers.pick(held, "c0ffee00") == readers.Picked(AGENT)


def test_a_name_a_reader_and_an_agent_share_refuses_and_marks_which_is_which():
    twin = _token("instinct", "d" * 64)
    got = readers.pick([LEO, AGENT, twin], "Instinct")
    assert got["status"] == "ambiguous_reader"
    assert [r.get("agent", False) for r in got["readers"]] == [True, False]


def test_a_name_nobody_holds_lists_the_agent_marked():
    got = readers.pick([LEO, AGENT], "Priya")
    assert got["status"] == "no_such_reader"
    assert [(r["label"], r.get("agent", False)) for r in got["readers"]] == [
        ("Leo", False), ("Instinct", True)]


def test_cutting_off_an_agent_says_it_can_no_longer_change_the_library():
    held = [LEO, MAYA, AGENT]
    before = readers.preview(AGENT, held, "https://svc.test")
    assert before["agent"] is True and before["readers_remaining"] == 2
    assert before["reader_label"] == "Instinct" and before["scope"] == "one_reader"
    said = " ".join(before["consent"])
    assert "an agent that acts as the user" in said and "or change it" in said
    assert "`to='agent'`" in said and "2 other(s) keep their access" in said

    after = readers.revoked(AGENT, held, True)
    assert after["status"] == "reader_revoked" and after["agent"] is True
    assert after["readers_remaining"] == 2 and "or change it" in after["message"]
    assert readers.revoked(AGENT, held, False)["status"] == "already_gone"


def test_cutting_off_a_reader_says_nothing_about_changing_the_library():
    held = [LEO, AGENT]
    before, after = readers.preview(LEO, held, "https://svc.test"), readers.revoked(LEO, held, True)
    assert "agent" not in before and "agent" not in after
    assert "change" not in " ".join(before["consent"]) + after["message"]
    # The one who keeps access here is an agent, so the count does not call it a reader.
    assert "1 other(s) keep their access" in before["consent"][0]
