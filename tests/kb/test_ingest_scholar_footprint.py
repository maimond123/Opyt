"""ingest_scholar_footprint — a scholar Oracle's papers into the TRUSTED store.

Offline: the adapter's `_get` is monkeypatched with byte literals and the breaker is injected.
`resolve_fulltext` is never stubbed because it must never be REACHED — `fulltext=None` is the
whole point of this path, and a test that stubs it could not tell the difference.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pipeline.kb import frontier_sources as fs
from pipeline.kb import ingest_papers as ip
from pipeline.kb import ingest_scholar_footprint as isf
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


class _PassThrough:
    def call(self, fn):
        return fn()

    def allow(self):
        return True


def _work(n, *, authors=None, doi=True, abstract=True):
    return {
        "id": f"https://openalex.org/W{n}",
        **({"doi": f"https://doi.org/10.1234/w{n}"} if doi else {}),
        "title": f"Work Number {n}",
        "publication_date": f"2026-0{(n % 9) + 1}-01",
        "abstract_inverted_index": ({"An": [0], "abstract": [1], f"w{n}": [2]}
                                    if abstract else None),
        "primary_location": {"source": {"display_name": "Nature"},
                             "landing_page_url": f"https://nature.com/w{n}"},
        "type": "article", "cited_by_count": n,
        "authorships": authors if authors is not None else [
            {"author": {"id": "https://openalex.org/A5000000001",
                        "display_name": "First Author"},
             "author_position": "first"},
            {"author": {"id": "https://openalex.org/A5043841592",
                        "display_name": "F. Arnold",
                        "orcid": "https://orcid.org/0000-0002-4027-364X"},
             "author_position": "last"}],
    }


def _adapter(monkeypatch, pages):
    """`pages` is a list of response dicts, served in order."""
    seen = {"urls": []}
    it = iter(pages)

    def _fake_get(url, *, headers=None):
        seen["urls"].append(url)
        return json.dumps(next(it)).encode()
    monkeypatch.setattr(fs, "_get", _fake_get)
    a = fs.OpenAlexWorksAdapter(breaker=_PassThrough())
    a._seen = seen
    return a


# ── the works walk ───────────────────────────────────────────────────────────────

def test_a_back_catalogue_pages_by_cursor_and_stops_on_a_short_page(monkeypatch):
    """`next_cursor` is SERVER-supplied. A short page is checked as well as the cursor because a
    server that keeps handing one back would otherwise page to the request cap on every call."""
    a = _adapter(monkeypatch, [
        {"results": [_work(i) for i in range(200)], "meta": {"next_cursor": "c2"}},
        {"results": [_work(i) for i in range(200, 210)], "meta": {"next_cursor": "c3"}},
    ])

    works = a.works("A5043841592")

    assert len(works) == 210
    assert "cursor=%2A" in a._seen["urls"][0]        # the first page opens the cursor with `*`
    assert "cursor=c2" in a._seen["urls"][1]


def test_a_single_page_request_skips_the_cursor_entirely(monkeypatch):
    """The common candidate-probe call is ONE request with no paging state."""
    a = _adapter(monkeypatch, [{"results": [_work(1)]}])

    a.works("A5043841592", limit=25)

    assert "cursor" not in a._seen["urls"][0]
    assert "per-page=25" in a._seen["urls"][0]


def test_the_window_is_sent_as_a_publication_date_filter(monkeypatch):
    a = _adapter(monkeypatch, [{"results": []}])

    a.works("A5043841592", limit=25, since=datetime(2026, 3, 1, tzinfo=timezone.utc))

    assert "from_publication_date%3A2026-03-01" in a._seen["urls"][0]


# ── the mint ─────────────────────────────────────────────────────────────────────

def test_papers_land_as_oracle_footprint_atoms_abstract_only(conn, fake_embedder, monkeypatch):
    """ABSTRACT-ONLY is the design. `fulltext=None` is a real resolved value meaning "no PDF, use
    the abstract" — distinct from `atomize_paper`'s `_UNSET` sentinel, which would send every
    paper through `resolve_fulltext` and its PDF mirrors."""
    def _boom(paper):
        raise AssertionError("resolve_fulltext must never be reached on this path")
    monkeypatch.setattr(ip, "resolve_fulltext", _boom)
    a = _adapter(monkeypatch, [{"results": [_work(1), _work(2)]}])

    out = isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592",
                                     limit=25, adapter=a)

    assert out["atoms"] == 2
    rows = conn.execute("SELECT entry_mode, payload FROM atoms ORDER BY atom_id").fetchall()
    assert {r[0] for r in rows} == {"oracle-footprint"}
    assert all(json.loads(r[1])["has_fulltext"] is False for r in rows)
    assert all(json.loads(r[1])["body_state"] == "partial" for r in rows)


def test_who_id_is_the_papers_own_first_author_never_the_oracle(conn, fake_embedder, monkeypatch):
    """The trust-laundering invariant, and the reason this arm needs no eligibility gate. A paper
    is multi-author by definition; attributing one to the Oracle who happens to co-write it is
    exactly what a gate would exist to stop, and `atomize_paper` refuses it at the source."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    a = _adapter(monkeypatch, [{"results": [_work(1)]}])

    isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592", limit=25, adapter=a)

    who = conn.execute("SELECT who_id FROM atoms").fetchone()[0]
    assert who != "openalex:A5043841592"             # NOT the Oracle whose feed this was
    assert who.startswith("paper-authors:")          # no S2 id resolved → the honest placeholder


def test_every_coauthor_reaches_the_atom_with_their_ids(conn, fake_embedder, monkeypatch):
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    a = _adapter(monkeypatch, [{"results": [_work(1)]}])

    isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592", limit=25, adapter=a)

    authors = json.loads(conn.execute("SELECT payload FROM atoms").fetchone()[0])["authors"]
    assert authors == [
        {"name": "First Author", "openalex_id": "A5000000001", "position": "first"},
        {"name": "F. Arnold", "openalex_id": "A5043841592",
         "orcid": "0000-0002-4027-364X", "position": "last"}]


def test_a_work_with_no_url_or_no_body_is_dropped_not_written(conn, fake_embedder, monkeypatch):
    """Papers are immutable under Policy B, so a contentless atom is frozen forever. A work with
    no DOI and no landing page also has no offline route to an atom id."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    no_url = {**_work(9, doi=False), "primary_location": {}}
    no_body = {**_work(8, abstract=False), "title": ""}
    a = _adapter(monkeypatch, [{"results": [_work(1), no_url, no_body]}])

    out = isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592",
                                     limit=25, adapter=a)

    assert out["fetched"] == 3 and out["papers"] == 1 and out["atoms"] == 1


def test_a_second_pull_re_mints_nothing(conn, fake_embedder, monkeypatch):
    """Policy B: an already-present paper is skipped BEFORE any work. Papers are immutable."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    works = {"results": [_work(1), _work(2)]}

    first = isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592", limit=25,
                                       adapter=_adapter(monkeypatch, [works]))
    again = isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592", limit=25,
                                       adapter=_adapter(monkeypatch, [works]))

    assert first["atoms"] == 2
    assert again["atoms"] == 0 and again["deduped"] == 2
    assert conn.execute("SELECT count(*) FROM atoms").fetchone()[0] == 2


def test_an_author_with_no_works_in_the_window_is_an_observation_not_a_failure(conn,
                                                                              fake_embedder,
                                                                              monkeypatch):
    a = _adapter(monkeypatch, [{"results": []}])

    out = isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592",
                                     limit=25, adapter=a)

    assert out == {"source": "openalex", "openalex_id": "A5043841592", "fetched": 0,
                   "atoms": 0, "papers": 0, "topics": None}


def test_an_open_breaker_raises_rather_than_reporting_an_empty_pull(conn, fake_embedder):
    """A host backing off is NOT an author who published nothing. Reporting empty would advance
    the cursor and buy one bad night a full TTL of silence."""
    class _Open(fs.OpenAlexWorksAdapter):
        def available(self):
            return False

    with pytest.raises(fs.SourceError):
        isf.sync_scholar_footprint(conn, fake_embedder, openalex_id="A5043841592", adapter=_Open())


def test_a_whole_pull_embeds_in_batches_not_once_per_paper(conn, recording_embedder,
                                                           monkeypatch):
    """`atomize_paper` is the only mint helper and every caller drives it one paper at a time.
    Without a shared sink, N papers cost N embed round-trips. `len(.calls)` counts FLUSHES."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    a = _adapter(monkeypatch, [{"results": [_work(i) for i in range(12)]}])

    out = isf.sync_scholar_footprint(conn, recording_embedder, openalex_id="A5043841592",
                                     limit=25, adapter=a)

    assert out["atoms"] == 12
    assert len(recording_embedder.calls) < 12


# ── the refresh dispatch ─────────────────────────────────────────────────────────

def test_an_openalex_entity_becomes_a_pullable_pair():
    """`source_key` is what the ADAPTER takes — the BARE author id, because that is what
    `/works?filter=author.id:` wants."""
    from pipeline.kb import oracle_refresh_state as st

    row = {"entity_id": "openalex:A5043841592", "identity_links": None, "profile": None}
    assert st.pair_from_member(row) == ("openalex", "A5043841592")


def test_a_scholar_only_entity_has_no_pullable_pair():
    """A Semantic Scholar author id has no works feed OPYT can pull. None, not a guess."""
    from pipeline.kb import oracle_refresh_state as st

    assert st.pair_from_member(
        {"entity_id": "scholar:2081297", "identity_links": None, "profile": None}) is None


# ── reachability through `_ingest_oracle` ────────────────────────────────────────
#
# The gap that let two bugs ship green on 2026-09-08: every test above drives
# `sync_scholar_footprint` DIRECTLY, so none of them noticed that the only production caller
# could never reach it. `_ingest_oracle` gated on `_root_profile`, which seeds discovery from an
# X / Substack / blog member only — so a researcher found purely through a saved paper returned
# "no rootable profile" and their papers were never pulled.

def _scholar_only_oracle(conn, openalex_id="A5043841592"):
    """A cluster carrying nothing but an OpenAlex author id — no X, no Substack, no blog."""
    schema.upsert_entity(conn, f"openalex:{openalex_id}", name="Frances H. Arnold",
                         identity_links=["https://orcid.org/0000-0002-4027-364X"])
    conn.execute("INSERT OR REPLACE INTO oracles(canonical_id) VALUES (?)",
                 (f"openalex:{openalex_id}",))
    conn.commit()
    return {"canonical_id": f"openalex:{openalex_id}", "name": "Frances H. Arnold"}


def test_a_scholar_only_oracle_still_gets_its_papers_pulled(conn, monkeypatch):
    """The regression test for the hoist. A cluster with no rootable profile must still reach the
    paper adapter — that Oracle is the whole reason the scholar rail exists."""
    from pipeline.kb import oracles

    oracle = _scholar_only_oracle(conn)
    calls = {}

    def fake_sync(c, embedder, *, openalex_id, author_name=None, since=None, topics=None):
        calls["openalex_id"] = openalex_id
        return {"papers": 28, "atoms": 28}

    monkeypatch.setattr(isf, "sync_scholar_footprint", fake_sync)
    res = oracles._ingest_oracle(conn, None, oracle)

    assert "error" not in res, res
    assert calls["openalex_id"] == "A5043841592"
    assert res["atoms_added"] == 28
    assert [(r["type"], r["action"]) for r in res["results"]] == [("openalex", "ingested")]
    # No root means no discovery ran — reported, not silently absent.
    assert res["discovery_ran_fresh"] is False


def test_an_oracle_with_neither_a_root_nor_an_author_id_is_still_refused(conn):
    """The gate did not disappear, it narrowed. Nothing to discover AND nothing to pull is still
    an error, because there is genuinely no work — as distinct from work we could not reach."""
    from pipeline.kb import oracles

    schema.upsert_entity(conn, "org:example.com", name="Example Org")
    conn.execute("INSERT OR REPLACE INTO oracles(canonical_id) VALUES ('org:example.com')")
    conn.commit()
    res = oracles._ingest_oracle(conn, None, {"canonical_id": "org:example.com",
                                              "name": "Example Org"})
    assert "no rootable profile" in res["error"]
    assert "no OpenAlex author id" in res["error"]


def test_a_scholar_only_oracle_seeds_no_trust_root(conn, monkeypatch):
    """David's 2026-09-08 ruling: a researcher is a topic-scoped feed, not a vouched voice. The
    trust root is seeded by the FOOTPRINT half, and a paper corpus discovers no accounts, so a
    root seeded from one would have nothing to verify."""
    from pipeline.ingestion import discover_profile as dp
    from pipeline.kb import oracles

    oracle = _scholar_only_oracle(conn)
    monkeypatch.setattr(isf, "sync_scholar_footprint",
                        lambda *a, **k: {"papers": 3, "atoms": 3})

    def explode(*a, **k):
        raise AssertionError("discover_profile must not run for a scholar-only Oracle")

    monkeypatch.setattr(dp, "discover_profile", explode)
    res = oracles._ingest_oracle(conn, None, oracle)
    assert res["atoms_added"] == 3


def test_a_scholar_canonical_id_is_a_canonical_reference_not_a_handle():
    """`openalex:` is the prefix every live scholar candidate carries. Missing from
    `_CANONICAL_PREFIX`, `add_oracle` classified it as a bare X @handle and sent it to
    `_fetch_x_identity`."""
    from pipeline.kb.oracles import _classify_reference

    assert _classify_reference("openalex:A5000946175") == "canonical"
    assert _classify_reference("scholar:2081297") == "canonical"
