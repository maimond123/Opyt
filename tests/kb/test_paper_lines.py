"""Paper records and paper lines.

A paper linked from a post you saved, or from an Oracle's X post, becomes a record (abstract only).
Every other linked paper gets one line in the post that links it, from its metadata: title, then
authors · year · venue. Blog, Substack and GitHub posts record their paper links when first built.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import (ingest_common, ingest_papers, ingest_x, link_cards, owed_links, paper_rule,
                         schema)
from pipeline.kb.link_cards import Card
from pipeline.oracles import roster
from pipeline.platforms.papers import landing
from pipeline.oracles.sources.references import state
from pipeline.oracles.sources.references.cards import Cards
from pipeline.oracles.sources.references.documents import Documents
from pipeline.platforms.web import page_head
from pipeline.platforms.web.page_head import Head
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.quota_loop.source import Retry

_PAPER_URL = "https://arxiv.org/abs/2401.00001"
_PAPER = {"paperId": "arXiv:2401.00001", "title": "Scaling Autonomous Agents",
          "abstract": "We study how autonomous agents compose small tools.",
          "authors": [{"name": "Alice Researcher"}, {"name": "Bob Coauthor"},
                      {"name": "Cy Third"}, {"name": "Di Fourth"}],
          "year": 2024, "venue": "NeurIPS", "url": _PAPER_URL,
          "externalIds": {"ArXiv": "2401.00001"}}
_LINE = Card("Scaling Autonomous Agents",
             "Alice Researcher, Bob Coauthor, Cy Third et al. · 2024 · NeurIPS")


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _loop(source):
    health = Outages(PLATFORM_PAUSES_S)
    return QuotaLoop("web", [source(health)], executor=Serial(), outages=health,
                     connect=schema.connect)


def _post(conn, atom_id, *, source_type, entry_mode, who_id, body):
    """A long-form post written through `store_atom`, which every KB atom passes, as an archive pull
    does."""
    url = "https://" + atom_id.split(":", 1)[1]
    md = (f"---\nsource: {source_type}\nurl: {url}\n---\n\n# A post\n\n{body}\n\n"
          f"---\n*Blog · [Original post]({url})*\n")
    raw_hash = ingest_common.snapshot_hash(md)
    ingest_common.store_atom(conn, {
        "atom_id": atom_id, "source_type": source_type, "what_kind": "opinion", "who_id": who_id,
        "when_ts": "2026-10-01", "when_precision": "day", "source_url": url,
        "raw_hash": raw_hash, "description": "A post", "entry_mode": entry_mode, "payload": {}},
        md)


def _payload(conn, atom_id):
    return json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                   (atom_id,)).fetchone()[0])


def _body(conn, atom_id):
    return schema.read_body(conn, atom_id)


def _metadata(monkeypatch, paper):
    calls = []
    monkeypatch.setattr(ingest_papers, "paper_from_url",
                        lambda url, **kw: calls.append(url) or (dict(paper) if paper else None))
    return calls


# ── the rule ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source_type, entry_mode, record", [
    ("x", "user-saved", True),               # an X bookmark
    ("substack", "user-saved", True),        # a saved Substack post, or one put in the hopper
    ("blog", "user-saved", True),            # an article put in the hopper
    ("x", "oracle-footprint", True),         # an Oracle's own X post
    ("blog", "oracle-footprint", False),     # an Oracle's essay: its papers are citations
    ("substack", "oracle-footprint", False),
    ("github", "oracle-footprint", False),   # a README
    ("blog", "author_referenced", False),    # a page an Oracle linked
    ("x", "author_referenced", False),       # an X post an Oracle linked
])
def test_who_earns_a_record(source_type, entry_mode, record):
    assert paper_rule.earns_record(source_type, entry_mode) is record


def _link(url, earns, post="p"):
    return state.Link(post, url, {}, Retry(), earns)


def test_a_paper_is_a_line_only_when_no_referring_post_earns_a_record():
    other = "https://doi.org/10.1038/s41586-024-07421-0"
    links = [_link(_PAPER_URL, False, "blog"), _link(_PAPER_URL, True, "x"),
             _link(other, False, "blog"), _link("https://a.com/essay", False, "blog")]
    assert state.line_papers(links) == {other}         # never a non-paper link


# ── long-form posts record their paper links, once, where every atom is written ──

def test_a_long_form_post_owes_its_paper_links_and_nothing_else():
    md = ("See [the paper](https://arxiv.org/abs/2401.00001), also "
          "https://doi.org/10.1038/s41586-024-07421-0. A [blog](https://a.com/post) and "
          "![fig](https://arxiv.org/abs/2402.00002)\n\n---\n*Blog · [Original post](https://b.com/p)*")
    atom = {"atom_id": "blog:b.com/p", "source_type": "blog", "entry_mode": "oracle-footprint",
            "who_id": "blog:b.com", "source_url": "https://b.com/p", "when_ts": "2026-10-01T09:00",
            "payload": {}}
    owed = owed_links.long_form_links(atom, md)
    assert [e["url"] for e in owed] == ["https://arxiv.org/abs/2401.00001",
                                        "https://doi.org/10.1038/s41586-024-07421-0"]
    assert owed[0]["ref"] == {"by": "blog:b.com", "handle": "", "post": "blog:b.com/p",
                              "url": "https://b.com/p", "date": "2026-10-01"}


@pytest.mark.parametrize("atom", [
    {"atom_id": "ssprobe:1", "source_type": "substack", "payload": {}},          # a candidate sample
    {"atom_id": "x:1", "source_type": "x", "entry_mode": "user-saved", "payload": {}},  # X: own rule
    {"atom_id": "blog:b.com/p", "source_type": "blog", "entry_mode": "oracle-footprint",
     "payload": {"links_owed": []}},                                              # already read
    {"atom_id": "blog:b.com/p", "source_type": "blog", "entry_mode": "oracle-footprint",
     "payload": "{}"},                                                            # a stored row
])
def test_atoms_that_owe_nothing_this_way(atom):
    assert owed_links.long_form_links(atom, f"see {_PAPER_URL}") is None


def test_a_post_written_through_store_atom_stores_what_it_owes(conn):
    _post(conn, "blog:b.com/p", source_type="blog", entry_mode="oracle-footprint",
          who_id="blog:b.com", body=f"See [the paper]({_PAPER_URL}).")
    assert [e["url"] for e in _payload(conn, "blog:b.com/p")["links_owed"]] == [_PAPER_URL]


# ── a line: an Oracle's essay links a paper ─────────────────────────────────────

def test_an_essays_paper_gets_a_line_in_the_post_and_no_record(conn, monkeypatch):
    roster.upsert_oracle(conn, "blog:b.com", name="Ben")
    _post(conn, "blog:b.com/p", source_type="blog", entry_mode="oracle-footprint",
          who_id="blog:b.com", body=f"See [the paper]({_PAPER_URL}).")
    conn.commit()
    calls = _metadata(monkeypatch, _PAPER)
    cards, documents = _loop(Cards), _loop(Documents)
    for loop in (cards, documents):
        loop.ctx.cache.update(session={})
    assert documents.sources[0].next_unit(conn, _now()) is None    # never a record

    assert cards.step(conn).result == "filled"

    assert calls == [_PAPER_URL]                                    # metadata, never the paper
    body = _body(conn, "blog:b.com/p")
    assert f"> **[Scaling Autonomous Agents]({_PAPER_URL})**" in body
    assert body.rstrip().endswith("*Blog · [Original post](https://b.com/p)*")   # footer last
    assert link_cards.stored(conn, _PAPER_URL) == _LINE
    assert _payload(conn, "blog:b.com/p")["links_owed"] == []
    assert conn.execute("SELECT count(*) FROM bodies_fts JOIN bodies b ON b.body_id = "
                        "bodies_fts.rowid WHERE b.atom_id='blog:b.com/p' "
                        "AND bodies_fts MATCH '\"Alice Researcher\"'").fetchone()[0] == 1  # searchable
    assert not schema.stored_atom_id(conn, "paper:arXiv:2401.00001")


def test_a_post_an_oracle_linked_gets_its_papers_line_and_no_record(conn, monkeypatch):
    norm = {"id": "200", "text": "new paper", "url": "https://x.com/dan/status/200",
            "createdAt": "Sat Aug 01 12:00:00 +0000 2026", "author": {"id": "2", "userName": "dan"},
            "entities": {"urls": [{"expanded_url": _PAPER_URL}]}}
    ingest_x.referenced_x_atom(conn, norm, f"# dan\n\nnew paper {_PAPER_URL}",
                               seen={})
    conn.commit()
    _metadata(monkeypatch, _PAPER)
    cards, documents = _loop(Cards), _loop(Documents)
    for loop in (cards, documents):
        loop.ctx.cache.update(session={})
    assert documents.sources[0].next_unit(conn, _now()) is None    # never a record

    assert cards.step(conn).result == "filled"

    assert f"> **[Scaling Autonomous Agents]({_PAPER_URL})**" in _body(conn, "x:200")
    assert _payload(conn, "x:200")["links_owed"] == []


def test_a_line_takes_its_year_from_the_date_when_no_year_is_given():
    assert link_cards.paper_line({"title": "T", "authors": [{"name": "A"}],
                                  "publicationDate": "2017-06-12"}) == Card("T", "A · 2017")


def test_a_paper_no_registry_names_is_named_by_its_page(monkeypatch):
    _metadata(monkeypatch, None)                     # a PubMed record that declares no DOI
    monkeypatch.setattr(page_head, "read_head", lambda url: Head("Aspirin and colorectal cancer",
                                                                 "A cohort study."))
    assert link_cards.read_paper("https://pubmed.ncbi.nlm.nih.gov/12345678/") == Card(
        "Aspirin and colorectal cancer", "A cohort study.")


_ECCC_URL = "https://eccc.weizmann.ac.il/report/2026/237/"
_ECCC_PAGE = {"title": "Counterexamples to Beyond-Johnson Proximity Gaps over Binary Fields",
              "authors": [{"name": "Quang Dao"}, {"name": "Scott Duke Kominers"},
                          {"name": "Justin Thaler"}, {"name": "Kai Zhe Zheng"}],
              "abstract": "Many hash-based proof systems check that committed words are close.",
              "year": 2026, "publicationDate": "2026-10-09",
              "venue": "Electronic Colloquium on Computational Complexity (ECCC)",
              "openAccessPdf": {"url": "https://eccc.weizmann.ac.il/report/2026/237/download"}}


def test_a_report_gets_its_line_from_its_landing_page(monkeypatch):
    """ECCC and IACR ePrint mint no DOI: the report's own page names it (`landing.metadata`)."""
    monkeypatch.setattr(landing, "metadata", lambda url: dict(_ECCC_PAGE))
    assert link_cards.read_paper(_ECCC_URL) == Card(
        "Counterexamples to Beyond-Johnson Proximity Gaps over Binary Fields",
        "Quang Dao, Scott Duke Kominers, Justin Thaler et al. · 2026 · "
        "Electronic Colloquium on Computational Complexity (ECCC)")


def test_a_refused_lookup_reads_nothing_and_the_link_stays_owed(monkeypatch):
    _metadata(monkeypatch, {**_PAPER, "title": None,
                            ingest_papers.S2_VERDICT: ingest_papers.FETCH_UNDETERMINED})
    monkeypatch.setattr(page_head, "read_head",
                        lambda url: pytest.fail("a throttled lookup must not fall back to the page"))
    assert link_cards.read_paper(_PAPER_URL) is None


def test_a_line_already_written_is_left_alone():
    md = "# A post\n\nSee it.\n\n---\n*Blog · [Original post](https://b.com/p)*\n"
    once = link_cards.fill_markdown(md, _PAPER_URL, _LINE)
    assert link_cards.fill_markdown(once, _PAPER_URL, _LINE) == once
    twice = link_cards.fill_markdown(once, "https://doi.org/10.1/x", Card("Second", ""))
    assert twice.count("## Links") == 1 and "> **[Second](https://doi.org/10.1/x)**" in twice


# ── a record: a saved Substack post links a paper ───────────────────────────────

def test_a_saved_posts_paper_becomes_an_abstract_only_record(conn, monkeypatch):
    _post(conn, "substack:datastream.substack.com/p/x", source_type="substack",
          entry_mode="user-saved", who_id="substack:datastream", body=f"Read {_PAPER_URL} today.")
    conn.commit()
    _metadata(monkeypatch, _PAPER)
    monkeypatch.setattr(ingest_papers, "resolve_fulltext",
                        lambda paper: pytest.fail("a record never reads its PDF at ingest"))
    cards, documents = _loop(Cards), _loop(Documents)
    for loop in (cards, documents):
        loop.ctx.cache.update(session={})
    assert cards.sources[0].next_unit(conn, _now()) is None          # never a line

    assert documents.step(conn).result == "captured"

    pay = _payload(conn, "paper:arXiv:2401.00001")
    assert pay["body_state"] == "partial" and pay["abstract"].startswith("We study")
    assert [r["post"] for r in pay["referenced_by"]] == ["substack:datastream.substack.com/p/x"]
    assert _payload(conn, "substack:datastream.substack.com/p/x")["links_owed"] == []


def test_a_saved_posts_report_becomes_a_record_from_its_page(conn, monkeypatch):
    """An ePrint or ECCC link has no DOI to identify it, so a report is keyed on its page and
    recorded from what the page declares."""
    _post(conn, "substack:datastream.substack.com/p/x", source_type="substack",
          entry_mode="user-saved", who_id="substack:datastream", body=f"Read {_ECCC_URL} today.")
    conn.commit()
    monkeypatch.setattr(landing, "metadata", lambda url: dict(_ECCC_PAGE))
    monkeypatch.setattr(ingest_papers, "resolve_fulltext",
                        lambda paper: pytest.fail("a record never reads its PDF at ingest"))
    documents = _loop(Documents)
    documents.ctx.cache.update(session={})

    assert documents.step(conn).result == "captured"

    pay = _payload(conn, "paper:url:eccc.weizmann.ac.il/report/2026/237")
    assert pay["title"] == _ECCC_PAGE["title"] and pay["abstract"] == _ECCC_PAGE["abstract"]
    assert pay["pdf_url"] == "https://eccc.weizmann.ac.il/report/2026/237/download"
    assert [r["post"] for r in pay["referenced_by"]] == ["substack:datastream.substack.com/p/x"]


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
