"""Substack samples (`oracles/sources/web/substack_samples.py`): a candidate with no X gets up to
five titles from one archive GET, and the X handle their primary byline declares; a failed GET
writes nothing."""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.oracles import probe_store, roster
from pipeline.oracles.sources.web import substack_samples as ss

from .._support import NOW

PUB = 1786017


def _post(pid, title, *, bylines=None):
    return {"id": pid, "title": title, "subtitle": f"{title}, explained",
            "post_date": "2026-09-02T02:43:22.397Z", "publication_id": PUB,
            "canonical_url": f"https://nine.substack.com/p/{pid}",
            "publishedBylines": bylines or []}


def _byline(handle, *, primary=True, pub=PUB):
    return {"twitter_screen_name": handle,
            "publicationUsers": [{"publication_id": pub, "is_primary": primary}]}


def test_titles_become_probe_atoms_newest_five():
    posts = [_post(i, f"Post {i}") for i in range(1, 8)]
    atoms = ss.sample_atoms(posts, "substack:nine")
    assert [a["atom_id"] for a in atoms] == [f"ssprobe:{i}" for i in range(1, 6)]
    assert atoms[0]["_markdown"] == "# Post 1\n\nPost 1, explained"
    assert {a["who_id"] for a in atoms} == {"substack:nine"}


def test_the_primary_bylines_handle_counts_and_a_guests_does_not():
    guest = _post(1, "a", bylines=[_byline("guesty", primary=False), _byline("@nine")])
    assert ss.primary_x_handle([guest]) == "nine"
    assert ss.primary_x_handle([_post(2, "b", bylines=[_byline("other", pub=5)])]) is None
    assert ss.primary_x_handle([_post(3, "c", bylines=[_byline(None)])]) is None


@pytest.fixture()
def candidate(conn):
    roster.upsert_oracle(conn, "x:user:1", name="One")
    schema.upsert_entity(conn, "substack:nine", name="Nine",
                         identity_links=["https://nine.substack.com"])
    schema.ensure_signal(conn, "substack:nine", "subscribe", "substack")
    schema.upsert_entity(conn, "x:user:5", name="Five", profile={"handle": "five"})
    schema.ensure_signal(conn, "x:user:5", "follow", "x")
    return conn


@pytest.fixture()
def archive(monkeypatch):
    served: list = []

    def fake(url, params, **kw):
        nxt = served.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(ss, "public_get_json", fake)
    return served


def test_only_candidates_with_no_x_are_queued(candidate):
    assert [(c["who_id"], c["url"]) for c in ss.queue(candidate)] == [
        ("substack:nine", "https://nine.substack.com")]


def test_a_sample_writes_titles_and_the_declared_handle_once(candidate, archive, web_loop):
    archive.append([_post(1, "On rollups", bylines=[_byline("NineWrites")]), _post(2, "Proofs")])
    src = ss.SubstackSamples()
    unit = src.next_unit(candidate, NOW)
    assert web_loop.run_unit(candidate, src, unit).result == "ok"
    assert probe_store.count_probe_atoms(candidate, "substack:nine") == 2
    assert probe_store.pull_states(candidate)["substack:nine"]["status"] == "ok"
    assert schema.get_entity(candidate, "substack:nine")["profile"].count("NineWrites") == 1
    assert src.next_unit(candidate, NOW) is None              # sampled once
    assert sorted(probe_store.texts_by_person(candidate)["substack:nine"]) == [
        "# On rollups\n\nOn rollups, explained", "# Proofs\n\nProofs, explained"]


def test_a_failed_get_writes_nothing_and_waits_to_retry(candidate, archive, web_loop):
    archive.append(RuntimeError("HTTP 403"))
    src = ss.SubstackSamples()
    web_loop.run_unit(candidate, src, src.next_unit(candidate, NOW))
    assert probe_store.count_probe_atoms(candidate, "substack:nine") == 0
    row = probe_store.pull_states(candidate)["substack:nine"]
    assert row["status"] == "failed" and row["retry_at"]
    assert src.next_unit(candidate, NOW) is None



def test_a_candidate_on_their_own_domain_is_their_own_site(candidate):
    """Its dead domain is charged to them and never counts as Substack failing."""
    assert ss.SubstackSamples().next_unit(candidate, NOW).item.own_site is False
    assert ss.own_site("substack", "https://signull.xyz") is True
    assert ss.own_site("substack", "https://nine.substack.com") is False
