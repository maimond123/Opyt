"""mentions — the candidates' third arm: people the Oracles' own posts quote, mention or link.

The failures that matter: a post tagging many accounts outranking a post about one person (the
protein query, where Bo Wang's affiliation tags won), an Oracle listed as its own candidate, and
the evidence carrying the quoted post instead of the Oracle's words.
"""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.oracles import mentions, roster
from pipeline.oracles.candidates import candidates_payload


def _engage(kind, handle, uid=None, words="w"):
    return {"kind": kind, "who": f"x:user:{uid}" if uid else None, "handle": handle,
            "words": words, "url": f"https://x.com/o/status/{handle}-{kind}"}


def _oracle(conn, uid, handle):
    schema.upsert_entity(conn, f"x:user:{uid}", name=handle.title(), profile={"handle": handle})
    conn.execute("UPDATE entities SET canonical_id = entity_id WHERE entity_id = ?",
                 (f"x:user:{uid}",))
    roster.upsert_oracle(conn, f"x:user:{uid}", handle.title())


def _posts(conn, posts):
    """`posts` = [(author uid, text, engages)] → Oracle footprint atoms through the real writer."""
    for i, (uid, text, engages) in enumerate(posts):
        store_atom(conn, {"atom_id": f"xprofile:{i}", "source_type": "x",
                          "who_id": f"x:user:{uid}", "entry_mode": "oracle-footprint",
                          "description": "d", "raw_hash": f"h{i}",
                          "when_ts": f"2026-10-0{i % 9 + 1}", "payload": {"engages": engages}},
                   text)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    _oracle(c, 100, "soren")
    _oracle(c, 200, "imas")
    yield c
    c.close()


def _search(conn, query, oracle=None, k=10):
    named = mentions.oracle_authors(conn, oracle)
    return mentions.search_mentions(conn, query, authors=named["authors"],
                                    exclude=named["exclude"], k=k, evidence=3)


@pytest.mark.parametrize("base,kind,n,expected", [
    (1.0, "quote", 1, 1.0),
    (1.0, "mention", 1, 0.5),
    (1.0, "link", 1, 0.75),
    (1.0, "quote", 4, 0.25),
    (0.5, "mention", 2, 0.125),
])
def test_credit(base, kind, n, expected):
    assert mentions.credit(base, kind, n) == pytest.approx(expected)


def test_boost_raises_without_gating():
    assert mentions.boost(1.0, 1) == 1.0 and mentions.boost(1.0, 3) == pytest.approx(1.5)


def test_a_post_naming_one_person_beats_a_post_tagging_many(conn):
    tags = [_engage("mention", f"org{i}", uid=str(i)) for i in range(12)]
    _posts(conn, [
        (100, "a crypto rollup proof with my coauthors", tags),
        (100, "a crypto rollup proof", [_engage("mention", "student", uid="50")]),
    ])
    rows = _search(conn, "crypto rollup proof")
    assert rows[0]["who_id"] == "x:user:50"


def test_evidence_is_the_oracles_words_and_says_it_is_about_them(conn):
    _posts(conn, [
        (100, "crypto rollup quoted text", [_engage("quote", "bob", "7", "This is the wrong take")])])
    [row] = _search(conn, "crypto rollup")
    ev = row["evidence"][0]
    assert (ev["oracle"], ev["kind"], ev["words"]) == ("soren", "quote", "This is the wrong take")
    assert "about this person, not by them" in ev["provenance"].lower()


def test_oracles_are_never_candidates(conn):
    _posts(conn, [
        (100, "crypto rollup", [_engage("quote", "imas", "200"), _engage("link", "Imas"),
                                _engage("mention", "bob", "7")])])
    assert [r["handle"] for r in _search(conn, "crypto")] == ["bob"]


def test_named_by_several_oracles_boosts_and_lists_them(conn):
    _posts(conn, [
        (100, "crypto rollup", [_engage("mention", "bob", "7")]),
        (200, "crypto rollup", [_engage("mention", "bob", "7")]),
        (100, "crypto rollup", [_engage("mention", "carol", "8")]),
    ])
    rows = _search(conn, "crypto rollup")
    assert rows[0]["handle"] == "bob" and rows[0]["oracles"] == ["imas", "soren"]


def test_one_oracle_filter_and_handle_only_links_fold_into_the_id(conn):
    _posts(conn, [
        (100, "crypto rollup", [_engage("link", "Bob")]),
        (100, "crypto proof", [_engage("mention", "bob", "7")]),
        (200, "crypto rollup", [_engage("mention", "carol", "8")]),
    ])
    rows = _search(conn, "crypto", oracle="soren")
    assert [(r["who_id"], len(r["evidence"])) for r in rows] == [("x:user:7", 2)]


def test_empty_query_ranks_by_engagement(conn):
    _posts(conn, [
        (200, "one", [_engage("quote", "bob", "7")]),
        (200, "two", [_engage("quote", "bob", "7")]),
        (200, "three", [_engage("mention", "carol", "8")]),
    ])
    assert [r["handle"] for r in _search(conn, "", oracle="imas")] == ["bob", "carol"]


def test_an_unknown_oracle_names_the_choices(conn):
    out = mentions.oracle_authors(conn, "nobody")
    assert "error" in out and out["oracles"] == ["imas", "soren"]


def test_candidates_payload_adds_the_arm_and_joins_known_candidates(conn, monkeypatch):
    from pipeline.oracles import ranking
    from tests.kb.test_candidate_search import _Cand

    _posts(conn, [
        (100, "crypto rollup", [_engage("quote", "bob", "7"), _engage("mention", "dana", "9")])])
    monkeypatch.setattr(ranking, "rank_candidates", lambda c, **kw: [_Cand("7")])

    out = candidates_payload(conn, "crypto rollup")
    by_handle = {r["handle"]: r for r in out["candidates"]}
    assert by_handle["c7"]["basis"] == "mentioned" and by_handle["c7"]["canonical_id"] == "p:7"
    assert by_handle["dana"]["canonical_id"] is None and by_handle["dana"]["distinct_signals"] == 0
    assert "mentioned" in out["basis_key"]

    assert all(r["basis"] != "mentioned"
               for r in candidates_payload(conn, "crypto",
                                           min_signals=2)["candidates"])
    assert "error" in candidates_payload(conn, "crypto", oracle="nobody")
    # No query: the plain listing stays the candidate list unless one Oracle is asked about.
    assert {r["basis"] for r in candidates_payload(conn, "")["candidates"]} <= {"probed",
                                                                                     "saved"}
    assert "dana" in {r["handle"] for r in candidates_payload(conn, "",
                                                              oracle="soren")["candidates"]}


def test_naming_an_oracle_lists_only_the_people_they_named(conn, monkeypatch):
    """The probed and saved arms cannot be narrowed to one Oracle; merged in, they filled 20 of 30
    rows with people a16zcrypto never named."""
    from pipeline.oracles import ranking
    from tests.kb.test_candidate_search import _Cand, _seed

    _seed(conn, probe=[("x:user:5", "crypto rollup proof")],
          trusted=[("x:user:6", "crypto rollup proof")])
    _posts(conn, [(100, "crypto rollup", [_engage("quote", "bob", "7")])])
    monkeypatch.setattr(ranking, "rank_candidates",
                        lambda c, **kw: [_Cand("5"), _Cand("6"), _Cand("7")])

    everyone = candidates_payload(conn, "crypto rollup")["candidates"]
    assert {r["basis"] for r in everyone} == {"probed", "saved", "mentioned"}
    for query in ("crypto rollup", ""):
        rows = candidates_payload(conn, query, oracle="soren")["candidates"]
        assert [(r["handle"], r["basis"]) for r in rows] == [("c7", "mentioned")]


def test_naming_an_oracle_lists_their_coauthors_even_above_one_signal(conn):
    schema.upsert_entity(conn, "openalex:A2", name="Colleague")
    schema.set_signal(conn, "openalex:A2", "coauthor", "openalex", count=4,
                      extra={"with": ["x:user:200"]})
    out = candidates_payload(conn, "", oracle="imas", min_signals=2)
    assert out["coauthors"] == [{"name": "Colleague", "canonical_id": "openalex:A2", "papers": 4}]
    assert candidates_payload(conn, "", oracle="soren")["coauthors"] == []
