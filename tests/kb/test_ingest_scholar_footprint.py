"""A scholar Oracle's papers into the TRUSTED store: the OpenAlex works walk, and one work → one
paper atom (`oracles/sources/archive/openalex._paper_from_work`, saved by the web scraper's archive).

Offline: the adapter's `_get` is monkeypatched with byte literals and the breaker is injected.
`resolve_fulltext` is never stubbed because it must never be REACHED — `fulltext=None` is the
whole point of this path, and a test that stubs it could not tell the difference.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pipeline.platforms.papers import openalex
from pipeline.platforms.papers import registry as paper_registry
from pipeline.platforms.errors import SourceError
from pipeline.kb import ingest_papers as ip
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.worker.quota_loop.source import Context
from pipeline.oracles.sources.archive.openalex import OpenAlex


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
    monkeypatch.setattr(paper_registry, "get", _fake_get)
    a = openalex.OpenAlexClient(breaker=_PassThrough())
    a._seen = seen
    return a


# ── the works read ───────────────────────────────────────────────────────────────

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


# ── the mint: one work → one paper atom (the web scraper's OpenAlex archive) ────────
#
# The rules these tests pin hold for the one pull: the archive saves one work per unit, through
# `_paper_from_work` and `atomize_paper` (`background_scrapers/web/archive/openalex.py`).

def _mint(conn, *works):
    """Each work through the archive's `document`, its held atom then written, as one archive
    run saves them."""
    held = []
    out = [OpenAlex().document(conn, {"url": "A5043841592"}, w, Context("web"),
                               lambda atom, md: held.append((atom, md)))
           for w in works]
    for atom, md in held:
        store_atom(conn, atom, md)
    return out


def test_papers_land_as_oracle_footprint_atoms_abstract_only(conn, monkeypatch):
    """ABSTRACT-ONLY is the design. `fulltext=None` is a real resolved value meaning "no PDF, use
    the abstract" — distinct from `atomize_paper`'s `_UNSET` sentinel, which would send every
    paper through `resolve_fulltext` and its PDF mirrors."""
    def _boom(paper):
        raise AssertionError("resolve_fulltext must never be reached on this path")
    monkeypatch.setattr(ip, "resolve_fulltext", _boom)

    _mint(conn, _work(1), _work(2))

    rows = conn.execute("SELECT entry_mode, payload FROM atoms ORDER BY atom_id").fetchall()
    assert len(rows) == 2 and {r[0] for r in rows} == {"oracle-footprint"}
    assert all(json.loads(r[1])["has_fulltext"] is False for r in rows)
    assert all(json.loads(r[1])["body_state"] == "partial" for r in rows)


def test_who_id_is_the_papers_own_first_author_never_the_oracle(conn, monkeypatch):
    """The trust-laundering invariant, and the reason this arm needs no eligibility gate. A paper
    is multi-author by definition; attributing one to the Oracle who happens to co-write it is
    exactly what a gate would exist to stop, and `atomize_paper` refuses it at the source."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)

    _mint(conn, _work(1))

    who = conn.execute("SELECT who_id FROM atoms").fetchone()[0]
    assert who != "openalex:A5043841592"             # NOT the Oracle whose feed this was
    assert who.startswith("paper-authors:")          # no S2 id resolved → the honest placeholder


def test_every_coauthor_reaches_the_atom_with_their_ids(conn, monkeypatch):
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)

    _mint(conn, _work(1))

    authors = json.loads(conn.execute("SELECT payload FROM atoms").fetchone()[0])["authors"]
    assert authors == [
        {"name": "First Author", "openalex_id": "A5000000001", "position": "first"},
        {"name": "F. Arnold", "openalex_id": "A5043841592",
         "orcid": "0000-0002-4027-364X", "position": "last"}]


def test_a_work_with_no_url_or_no_body_is_dropped_not_written(conn, monkeypatch):
    """Papers are immutable under Policy B, so a contentless atom is frozen forever. A work with
    no DOI and no landing page also has no offline route to an atom id."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    no_url = {**_work(9, doi=False), "primary_location": {}}
    no_body = {**_work(8, abstract=False), "title": ""}

    out = _mint(conn, _work(1), no_url, no_body)

    assert out[1:] == ["skipped", "skipped"]
    assert conn.execute("SELECT count(*) FROM atoms").fetchone()[0] == 1


def test_an_open_breaker_raises_rather_than_reporting_an_empty_page():
    """A host backing off is NOT an author who published nothing. Reporting an empty page would
    finish the listing and buy one bad night a full TTL of silence."""
    class _Open(openalex.OpenAlexClient):
        def available(self):
            return False

    ctx = Context("web")
    ctx.cache["openalex"] = _Open()
    with pytest.raises(SourceError):
        OpenAlex().page({"url": "A5043841592"}, {"cursor": "*"}, ctx)


# ── the refresh dispatch ─────────────────────────────────────────────────────────

def test_an_openalex_entity_becomes_a_pullable_pair():
    """`source_key` is what the ADAPTER takes — the BARE author id, because that is what
    `/works?filter=author.id:` wants."""
    from pipeline.oracles import pairs

    row = {"entity_id": "openalex:A5043841592", "identity_links": None, "profile": None}
    assert pairs.pair_from_member(row) == ("openalex", "A5043841592")


def test_a_scholar_only_entity_has_no_pullable_pair():
    """A Semantic Scholar author id has no works feed OPYT can pull. None, not a guess."""
    from pipeline.oracles import pairs

    assert pairs.pair_from_member(
        {"entity_id": "scholar:2081297", "identity_links": None, "profile": None}) is None


# ── reachability ───────────────────────────────────────────────────────────────────
#
# A researcher found purely through a saved paper must still be pulled, with no root profile. The
# background scrapers pull an `openalex` pair without discovery (`web/archives.py`).

def test_a_scholar_canonical_id_is_a_canonical_reference_not_a_handle():
    """`openalex:` is the prefix every live scholar candidate carries. Missing from
    `_CANONICAL_PREFIX`, `add_oracle` classified it as a bare X @handle and sent it to
    `_fetch_x_identity`."""
    from pipeline.oracles.reference import classify_reference

    assert classify_reference("openalex:A5000946175") == "canonical"
    assert classify_reference("scholar:2081297") == "canonical"


def test_a_paper_the_oracle_linked_becomes_theirs_when_their_works_list_has_it(conn):
    """Stored `author_referenced` because they linked it, it was `present` and counted for no one.
    Their works list confirms authorship: it becomes footprint, with OpenAlex's author ids."""
    (aid,) = _mint(conn, _work(1))
    conn.execute("UPDATE atoms SET entry_mode='author_referenced', "
                 "payload=json_set(payload, '$.authors', json('[{\"name\": \"F. Arnold\"}]')) "
                 "WHERE atom_id=?", (aid,))
    conn.commit()
    assert _mint(conn, _work(1)) == [aid]
    mode, payload = conn.execute("SELECT entry_mode, payload FROM atoms WHERE atom_id=?",
                                 (aid,)).fetchone()
    assert mode == "oracle-footprint"
    assert [a.get("openalex_id") for a in json.loads(payload)["authors"]] == [
        "A5000000001", "A5043841592"]


def test_a_saved_paper_stays_saved_when_an_oracles_works_list_has_it(conn):
    (aid,) = _mint(conn, _work(1))
    conn.execute("UPDATE atoms SET entry_mode='user-saved' WHERE atom_id=?", (aid,))
    conn.commit()
    assert _mint(conn, _work(1)) == ["present"]
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id=?", (aid,)).fetchone()[0] == \
        "user-saved"
