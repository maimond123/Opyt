"""ingest_papers (shared paper → atom core) — offline wiring proof.

`resolve_fulltext` (the network + PDF step) is monkeypatched, so these prove the WIRING —
full-document chunk-embed, `paper:{canonical_id}` dedup keyed atom, `who_id` = the PAPER's
author (never the Oracle), the caller-supplied Oracle vouch + co-authorship + external-ref edges,
policy-B skip-before-fetch, and the abstract-only fail-safe — not the live PDF download.

`paper_from_url` normalization is tested with `enrich=False` (pure URL parsing, no S2 call).
"""
from __future__ import annotations

import pytest

from pipeline.kb import ingest_papers as ip
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


_PAPER = {
    "paperId": "arXiv:2401.00001",
    "title": "Scaling Autonomous Agents",
    "abstract": "We study how autonomous agents compose small tools into larger systems.",
    "authors": [
        {"authorId": "111", "name": "Alice Researcher"},
        {"authorId": "222", "name": "Bob Coauthor"},
        {"authorId": None, "name": "Nameless Contributor"},   # no id → no co-authorship edge
    ],
    "year": 2024,
    "publicationDate": "2024-01-15",
    "venue": "NeurIPS",
    "citationCount": 42,
    "url": "https://www.semanticscholar.org/paper/abc",
    "externalIds": {"ArXiv": "2401.00001", "DOI": "10.1234/abcd"},
    "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.00001"},
}
_ATOM = "paper:arXiv:2401.00001"
# A body that is UNMISTAKABLY the full document, not the abstract — so we can prove the full text
# (not just the abstract) landed in chunks.
_FULLTEXT = ("INTRODUCTION. This is the full body of the paper, far longer than any abstract. "
             "We describe the framework, the experiments, and the retry and timeout tradeoffs. ") * 30


def _stub_fulltext(monkeypatch, text):
    """Replace the network + PDF step with a deterministic return (text or None)."""
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: text)


class _FakeWorksAdapter:
    """The adapter seam, stubbed at `frontier_sources.OpenAlexWorksAdapter` — the name
    `_openalex_metadata` imports inside the function. No live network in this suite: a real
    `atomize_paper` once spent 44 seconds pulling a PDF for a fake arXiv id (tests/conftest.py)."""

    def __init__(self, work=None, *, up=True, raises=None, calls=None,
                 by_id=None, by_landing=None):
        self._work, self._up, self._raises, self._calls = work, up, raises, calls
        self._by_id, self._by_landing = by_id, by_landing or []

    def available(self):
        return self._up

    def work_by_doi(self, doi):
        if self._calls is not None:
            self._calls.append(doi)
        if self._raises is not None:
            raise self._raises
        return self._work

    # the other two reads on this adapter, so the fake is a COMPLETE stand-in — patching the class
    # away and leaving a method behind is how a test starts passing for the wrong reason
    def work_by_openalex_id(self, work_id):
        if self._raises is not None:
            raise self._raises
        return self._by_id

    def works_by_landing_page(self, url):
        if self._raises is not None:
            raise self._raises
        return list(self._by_landing)


def _openalex(monkeypatch, adapter):
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources, "OpenAlexWorksAdapter", lambda *a, **kw: adapter)


# ── the happy path: one paper → one FULL-BODY artifact atom ───────────────────────

def test_atomize_builds_full_body_atom(conn, fake_embedder, monkeypatch):
    _stub_fulltext(monkeypatch, _FULLTEXT)
    out = ip.atomize_paper(conn, fake_embedder, _PAPER)
    assert out == _ATOM

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()
    assert atom is not None
    assert atom["source_type"] == "paper"
    assert atom["what_kind"] == "artifact"                # a research artifact, not a hot take
    assert atom["entry_mode"] == "author_referenced"      # Oracle referenced, didn't author
    assert atom["who_id"] == "scholar:111"                # the PAPER's first author — NOT the Oracle
    assert atom["when_ts"] == "2024-01-15"

    import json
    payload = json.loads(atom["payload"])
    assert payload["has_fulltext"] is True
    assert payload["venue"] == "NeurIPS" and payload["citationCount"] == 42

    # THE core assertion: the FULL body (not just the abstract) is chunked + searchable, and no
    # frontmatter chrome leaked into the routing text.
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id=? ORDER BY seq", (_ATOM,)))
    assert "This is the full body of the paper" in body      # the full text landed
    assert "compose small tools into larger systems" in body  # the abstract too
    assert "source: paper" not in body                        # frontmatter stripped


# ── policy B: an immutable paper is skipped BEFORE any (paid) re-fetch ─────────────

def test_atomize_idempotent_policy_b(conn, fake_embedder, monkeypatch):
    _stub_fulltext(monkeypatch, _FULLTEXT)
    assert ip.atomize_paper(conn, fake_embedder, _PAPER) == _ATOM

    # Second run: resolve_fulltext must NOT be called again (policy B skips before the fetch).
    def _boom(paper):
        raise AssertionError("fulltext re-resolved for an already-ingested (immutable) paper")

    monkeypatch.setattr(ip, "resolve_fulltext", _boom)
    assert ip.atomize_paper(conn, fake_embedder, _PAPER) is None
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()[0] == 1


def test_atomize_seen_threading_dedups_across_a_batch(conn, fake_embedder, monkeypatch):
    # A driver threads `seen` across a run; the second call dedups without touching the DB check.
    _stub_fulltext(monkeypatch, _FULLTEXT)
    seen = schema.load_hashes(conn, "paper")
    assert ip.atomize_paper(conn, fake_embedder, _PAPER, seen=seen) == _ATOM
    assert _ATOM in seen
    assert ip.atomize_paper(conn, fake_embedder, _PAPER, seen=seen) is None


# ── fail-safe: no full text → abstract-only atom, never a crash ───────────────────

def test_atomize_failsafe_abstract_only(conn, fake_embedder, monkeypatch):
    _stub_fulltext(monkeypatch, None)     # no OA PDF / extraction failed
    out = ip.atomize_paper(conn, fake_embedder, _PAPER)
    assert out == _ATOM                   # atom STILL created (degrade, don't crash)

    import json
    payload = json.loads(
        conn.execute("SELECT payload FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()["payload"])
    assert payload["has_fulltext"] is False

    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id=?", (_ATOM,)))
    assert "compose small tools into larger systems" in body   # the abstract is the body
    assert "This is the full body of the paper" not in body    # there is no full text


# ── unidentifiable paper → None (never mints paper:None) ──────────────────────────

def test_atomize_none_on_unidentifiable_paper(conn, fake_embedder, monkeypatch):
    _stub_fulltext(monkeypatch, _FULLTEXT)
    assert ip.atomize_paper(conn, fake_embedder, {"title": "no ids here", "authors": []}) is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── the S2 metadata fetch verdict, and the skip it enables ───────────────────────
# REGRESSION SET. Until 2026-08-03 `_fetch_s2_paper` collapsed {answered, 404, 429, transport
# failure} into one bare None, and this whole seam had NO test — which is how a live 429 came to
# be stored as `body_state: absent`, i.e. "this paper has no abstract." Measured: two 429s then a
# 200 carrying a 1136-char abstract for arXiv:1706.03762. Unauthenticated S2 allows ~1 req/sec,
# so this was the common path.

class _Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def _stub_s2(monkeypatch, resp_or_exc):
    """Patch the `requests.get` that `_fetch_s2_paper` imports at call time, and neutralize its
    retry sleep. Returns the list of calls, so a test can assert HOW MANY requests were made —
    which is the only way to tell "retried and gave up" from "asked once".

    `resp_or_exc` may be a single response (returned every time) or a LIST consumed in order,
    which is what an intermittent 429 looks like from the caller's side.
    """
    import requests
    calls = []
    seq = list(resp_or_exc) if isinstance(resp_or_exc, list) else None

    def _get(url, **kw):
        calls.append(url)
        item = seq[min(len(calls) - 1, len(seq) - 1)] if seq else resp_or_exc
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(requests, "get", _get)
    monkeypatch.setattr(ip.time, "sleep", lambda _s: None)
    return calls


@pytest.mark.parametrize("resp,expect_data,expect_verdict", [
    (_Resp(200, {"title": "T", "abstract": "A"}), True, ip.FETCH_OK),
    (_Resp(404), False, ip.FETCH_ABSENT),           # S2 truly has no such paper — retry is pointless
    (_Resp(429), False, ip.FETCH_UNDETERMINED),     # THE case that caused the bug
    (_Resp(503), False, ip.FETCH_UNDETERMINED),
    (_Resp(200, ["not", "a", "dict"]), False, ip.FETCH_UNDETERMINED),
])
def test_s2_fetch_reports_a_verdict_not_just_none(monkeypatch, resp, expect_data, expect_verdict):
    _stub_s2(monkeypatch, resp)
    data, verdict = ip._fetch_s2_paper("arXiv:1706.03762")
    assert verdict == expect_verdict
    assert (data is not None) is expect_data


def test_s2_fetch_transport_failure_is_undetermined_not_absent(monkeypatch):
    """A dead network is indistinguishable from a block, and BOTH are retryable. Calling either
    one `absent` is the conflation this contract exists to prevent."""
    _stub_s2(monkeypatch, RuntimeError("connection reset"))
    data, verdict = ip._fetch_s2_paper("arXiv:1706.03762")
    assert (data, verdict) == (None, ip.FETCH_UNDETERMINED)


# ── PubMed: the one paper id that needs a lookup ────────────────────────────────
#
# `pubmed.ncbi.nlm.nih.gov` sat in `link_router._PAPER_HOSTS` with no branch in `_parse_paper_url`
# from the day the list was written, so every PubMed url routed to the paper adapter and returned
# `failed` — advertised as supported, refusing every link, and silent because a failed mint writes
# nothing at all.


def _stub_eutils(monkeypatch, doi):
    """NCBI E-utilities answering with (or without) a DOI for the PMID."""
    import requests
    calls = []

    def _get(url, **kw):
        calls.append(url)
        ids = [{"idtype": "pubmed", "value": "34265844"}]
        if doi:
            ids.append({"idtype": "doi", "value": doi})
        return _Resp(200, {"result": {"34265844": {"articleids": ids}}})
    monkeypatch.setattr(requests, "get", _get)
    return calls


def test_a_pubmed_url_mints_the_same_atom_as_its_doi(monkeypatch):
    """The identity is the point. A `pubmed:{pmid}` key would have split one immutable paper
    across two atoms depending on which url the user happened to paste."""
    _stub_eutils(monkeypatch, "10.1038/s41586-021-03819-2")
    _openalex(monkeypatch, _FakeWorksAdapter(None))   # the title fallback, silenced: id only here

    p = ip.paper_from_url("https://pubmed.ncbi.nlm.nih.gov/34265844/", enrich=True)

    assert p["paperId"] == "DOI:10.1038/s41586-021-03819-2"
    assert p["url"] == "https://pubmed.ncbi.nlm.nih.gov/34265844/", \
        "the url the USER saved is kept, as every other branch keeps theirs"


def test_the_offline_path_stays_offline(monkeypatch):
    """`enrich=False` means "no network", and `link_router.predicted_atom_id` rests on it — that
    is Hopper's free already-present pre-check, run once per candidate across a page of search
    results. A lookup here would turn scanning ten of them into ten round trips."""
    calls = _stub_eutils(monkeypatch, "10.1038/s41586-021-03819-2")

    assert ip.paper_from_url("https://pubmed.ncbi.nlm.nih.gov/34265844/", enrich=False) is None
    assert calls == []


def test_a_pubmed_record_with_no_doi_refuses_rather_than_guessing(monkeypatch):
    """Fail-safe, landing exactly on the OLD behaviour (unparseable → the caller reports failed)
    rather than on an invented id. Some PubMed records genuinely carry no DOI."""
    _stub_eutils(monkeypatch, None)

    assert ip.paper_from_url("https://pubmed.ncbi.nlm.nih.gov/34265844/", enrich=True) is None


def test_a_dead_eutils_never_raises(monkeypatch):
    """Enrichment is a bonus everywhere else in this module and it stays one here."""
    _stub_s2(monkeypatch, RuntimeError("dns failure"))

    assert ip.paper_from_url("https://pubmed.ncbi.nlm.nih.gov/34265844/", enrich=True) is None


# ── the retry (added 2026-09-09) ────────────────────────────────────────────────
#
# A 429 says the SHARED unauthenticated pool is busy, not that this caller is over a quota, and
# the contention is time-varying: measured minutes apart on the same endpoint, a 20-request burst
# took 16 429s and a later 12-paper batch took none. One request was a coin flip whose outcome is
# permanent — a paper that lands without a title can never be repaired under policy-B dedup.


def test_a_429_then_an_answer_is_an_answer(monkeypatch):
    """THE regression. `FETCH_UNDETERMINED` has always meant "we were stopped, retry me" and
    nothing ever did, so a transient block minted a permanently anonymous atom."""
    calls = _stub_s2(monkeypatch, [_Resp(429), _Resp(429), _Resp(200, {"title": "T"})])

    data, verdict = ip._fetch_s2_paper("arXiv:1706.03762")

    assert verdict == ip.FETCH_OK and data == {"title": "T"}
    assert len(calls) == 3


def test_a_404_is_never_retried(monkeypatch):
    """A 404 is an ANSWER — S2 has no record of this paper. Asking again returns the same 404 and
    spends a request from the pool everyone shares, making the busy window worse for the calls
    that could actually be rescued."""
    calls = _stub_s2(monkeypatch, _Resp(404))

    assert ip._fetch_s2_paper("arXiv:1706.03762") == (None, ip.FETCH_ABSENT)
    assert len(calls) == 1


def test_an_answer_costs_exactly_one_request(monkeypatch):
    """The retry must be free in the quiet window. If it were not, every paper ingest would pay
    for a problem that fires on a minority of calls."""
    calls = _stub_s2(monkeypatch, _Resp(200, {"title": "T"}))

    ip._fetch_s2_paper("arXiv:1706.03762")

    assert len(calls) == 1


def test_it_gives_up_and_says_so(monkeypatch):
    """Bounded, and still honest at the end. A sustained block must report UNDETERMINED rather
    than degrade to ABSENT — `frontier_admit` reads that verdict to decide what is worth another
    pass, so collapsing the two would retire a paper that was only ever throttled."""
    calls = _stub_s2(monkeypatch, _Resp(429))

    assert ip._fetch_s2_paper("arXiv:1706.03762") == (None, ip.FETCH_UNDETERMINED)
    assert len(calls) == ip._S2_RETRIES


def test_a_transport_failure_is_retried_too(monkeypatch):
    """A dead socket is indistinguishable from a block from here, and the existing contract
    already calls both retryable. It has to actually retry for that to mean anything."""
    calls = _stub_s2(monkeypatch, [RuntimeError("connection reset"), _Resp(200, {"title": "T"})])

    data, verdict = ip._fetch_s2_paper("arXiv:1706.03762")

    assert verdict == ip.FETCH_OK and data == {"title": "T"}
    assert len(calls) == 2


def test_the_last_attempt_is_not_followed_by_a_sleep(monkeypatch):
    """A caller about to be told UNDETERMINED must not also wait for the privilege. With
    `_S2_RETRIES` attempts there are `_S2_RETRIES - 1` gaps, never one per attempt."""
    import requests
    slept = []
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(429))
    monkeypatch.setattr(ip.time, "sleep", lambda s: slept.append(s))

    ip._fetch_s2_paper("arXiv:1706.03762")

    assert len(slept) == ip._S2_RETRIES - 1


def test_s2_fetch_without_a_lookup_id_is_absent_not_undetermined():
    """No lookup id = no route to S2 at all (a bare .pdf link). Nothing was blocked, so retrying
    changes nothing — marking it undetermined would skip the atom forever."""
    assert ip._fetch_s2_paper(None) == (None, ip.FETCH_ABSENT)


def test_paper_from_url_stamps_the_verdict(monkeypatch):
    _stub_s2(monkeypatch, _Resp(429))
    blocked = ip.paper_from_url("https://arxiv.org/abs/2401.00001")
    assert blocked[ip._S2_VERDICT] == ip.FETCH_UNDETERMINED

    _stub_s2(monkeypatch, _Resp(200, {"title": "T", "abstract": "A"}))
    ok = ip.paper_from_url("https://arxiv.org/abs/2401.00001")
    assert ok[ip._S2_VERDICT] == ip.FETCH_OK


def test_enrich_false_is_not_blocked():
    """`enrich=False` means the caller never asked, which is NOT the same as being stopped.
    Marking it undetermined would make every offline-normalized paper unatomizable."""
    p = ip.paper_from_url("https://arxiv.org/abs/2401.00001", enrich=False)
    assert p[ip._S2_VERDICT] == ip.FETCH_OK


def test_blocked_metadata_and_no_fulltext_writes_NOTHING(conn, fake_embedder, monkeypatch):
    """THE regression test. Before the skip this wrote a titleless atom marked `absent` — and
    Policy B (skip by atom-id existence, before any fetch) made that permanent, so one 429 froze
    a real paper as a contentless stub forever. Nothing written is what lets the next run retry."""
    _stub_fulltext(monkeypatch, None)
    blocked = {**_PAPER, "title": None, "abstract": None, ip._S2_VERDICT: ip.FETCH_UNDETERMINED}

    assert ip.atomize_paper(conn, fake_embedder, blocked) is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    # nothing marked processed either — Policy B's presence check must MISS next run
    assert ip._atom_exists(conn, _ATOM) is False


def test_blocked_metadata_still_writes_when_the_pdf_resolved(conn, fake_embedder, monkeypatch):
    """The deliberate NON-skip. The PDF is the body, so the atom is genuinely `complete` and only
    its metadata is thin. Skipping these would stall paper ingest whenever S2 is hot — trading
    wrong data for NO data on the one source whose body we can actually read."""
    import json
    _stub_fulltext(monkeypatch, _FULLTEXT)
    blocked = {**_PAPER, ip._S2_VERDICT: ip.FETCH_UNDETERMINED}

    assert ip.atomize_paper(conn, fake_embedder, blocked) == _ATOM
    payload = json.loads(
        conn.execute("SELECT payload FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()["payload"])
    assert payload["body_state"] == "complete"


@pytest.mark.parametrize("verdict", [ip.FETCH_OK, ip.FETCH_ABSENT, ip.FETCH_UNDETERMINED])
def test_no_body_is_skipped_whatever_the_verdict(conn, fake_embedder, monkeypatch, verdict):
    """The skip asks "is there a body?", never "why is there no body?" — for all three verdicts.

    Keyed on the verdict it missed `FETCH_ABSENT`, an S2 404, which fell through and wrote
    `body_state=absent, body_basis=observed` — "WE determined this paper has no body" — when what
    happened is that the one provider we asked had never heard of it. A false OBSERVED claim,
    permanent under Policy B. Not a corner: S2 resolved 1 of 15 OpenAlex DOIs on 2026-08-26, so
    that was the common path for every non-arXiv paper source.

    `FETCH_OK` is here deliberately. An S2 record that answers with `abstract: null` and has no
    PDF used to mint a legitimately-`absent` atom; it no longer does, because a title-only atom
    is still an atom with nothing to read in it.
    """
    _stub_fulltext(monkeypatch, None)
    paper = {**_PAPER, "abstract": None, ip._S2_VERDICT: verdict}

    assert ip.atomize_paper(conn, fake_embedder, paper) is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    # nothing marked processed either — Policy B's presence check must MISS next run
    assert ip._atom_exists(conn, _ATOM) is False


def test_a_written_paper_always_carries_a_body(conn, fake_embedder, monkeypatch):
    """The other half: `complete` and `partial` are the ONLY two states a paper atom can carry,
    because the no-body skip returns before the payload is built. `absent` is unreachable here —
    there is no longer a way to store a paper with nothing in it."""
    import json
    _stub_fulltext(monkeypatch, None)

    assert ip.atomize_paper(conn, fake_embedder, {**_PAPER}) == _ATOM      # abstract, no PDF
    payload = json.loads(
        conn.execute("SELECT payload FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()["payload"])
    assert payload["body_state"] == "partial"


def test_a_paper_dict_from_elsewhere_is_not_treated_as_blocked(conn, fake_embedder, monkeypatch):
    """`atomize_paper` is source-agnostic and takes Paper dicts that never went through
    `paper_from_url` (an S2 search result already carries its metadata). A MISSING verdict must
    read as "not blocked" — the opposite default would silently skip every such paper."""
    _stub_fulltext(monkeypatch, None)
    assert ip._S2_VERDICT not in _PAPER
    assert ip.atomize_paper(conn, fake_embedder, _PAPER) == _ATOM


# ── paper_from_url normalization (offline: enrich=False, pure URL parsing) ─────────

def test_paper_from_url_normalizes_arxiv():
    for u in ("https://arxiv.org/abs/2401.00001",
              "https://arxiv.org/pdf/2401.00001v2",
              "https://arxiv.org/pdf/2401.00001.pdf",
              "http://arxiv.org/abs/2401.00001v11"):
        assert ip.paper_from_url(u, enrich=False)["paperId"] == "arXiv:2401.00001"


def test_paper_from_url_normalizes_doi_and_s2():
    d = ip.paper_from_url("https://doi.org/10.1234/AbCd", enrich=False)
    assert d["paperId"] == "DOI:10.1234/abcd"                       # lowercased (case-insensitive)
    # A real S2 paper page carries a 40-hex hash, often behind a title slug.
    s2_id = "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567"
    s = ip.paper_from_url(f"https://www.semanticscholar.org/paper/Attention/{s2_id}", enrich=False)
    assert s["paperId"] == s2_id
    # A numeric CorpusId also resolves.
    n = ip.paper_from_url("https://www.semanticscholar.org/paper/2194775", enrich=False)
    assert n["paperId"] == "2194775"


def test_paper_from_url_generic_pdf_keeps_url_as_fulltext_source():
    pdf = ip.paper_from_url("https://example.com/papers/cool.pdf", enrich=False)
    assert pdf["paperId"].startswith("url:")
    assert pdf["openAccessPdf"]["url"] == "https://example.com/papers/cool.pdf"


def test_paper_from_url_returns_none_on_non_paper():
    assert ip.paper_from_url("https://twitter.com/someone/status/123", enrich=False) is None
    assert ip.paper_from_url("https://example.com/blog/post", enrich=False) is None
    assert ip.paper_from_url("", enrich=False) is None


# ── content_type hint: a raw PDF whose url gives no `.pdf` shape ───────────────────
# `link_router.classify_link_deep` already fetched the url and knows its REAL Content-Type — the
# one fact a bare url string can never assert about itself. Without it, a `/download?id=123`-style
# redirect that serves a pdf is indistinguishable from any other unrecognized link.

def test_paper_from_url_pdf_content_type_recovers_a_hintless_path():
    p = ip.paper_from_url("https://example.com/download?id=123", enrich=False,
                          content_type="application/pdf; charset=binary")
    assert p is not None and p["paperId"].startswith("url:")
    assert p["openAccessPdf"]["url"] == "https://example.com/download?id=123"


def test_paper_from_url_pdf_content_type_keys_on_the_full_url_not_just_the_path():
    # Two different papers behind the SAME generic download path, distinguished only by the query
    # string — collapsing to path-only (as the `.pdf`-suffix branch deliberately does, to shed
    # tracking params) would silently dedupe them into one atom. The full url must stay the key.
    a = ip.paper_from_url("https://example.com/download?id=1", enrich=False,
                          content_type="application/pdf")
    b = ip.paper_from_url("https://example.com/download?id=2", enrich=False,
                          content_type="application/pdf")
    assert a["paperId"] != b["paperId"]


def test_paper_from_url_content_type_hint_ignored_without_pdf_signal():
    # A hint that ISN'T a pdf content-type must not turn an unrecognized url into a paper — the
    # hint only ever narrows an existing "not a paper" answer to "yes, for this one extra reason".
    assert ip.paper_from_url("https://example.com/blog/post", enrich=False,
                             content_type="text/html") is None


# ── resolve_fulltext source ordering (offline: arXiv PDF preferred over OA link) ──

def test_fulltext_pdf_url_ordering():
    urls = ip._fulltext_pdf_urls(_PAPER)
    assert urls[0] == "https://arxiv.org/pdf/2401.00001"           # arXiv open mirror first
    assert "https://arxiv.org/pdf/2401.00001" in urls
    # A paper with only an OA link (no arXiv id) falls through to it.
    only_oa = {"externalIds": {}, "openAccessPdf": {"url": "https://ex.com/oa.pdf"}}
    assert ip._fulltext_pdf_urls(only_oa) == ["https://ex.com/oa.pdf"]
    # No open source at all → empty list → resolve_fulltext returns None (abstract-only).
    assert ip._fulltext_pdf_urls({"externalIds": {}}) == []


# ── Zenodo metadata fallback: the source's own record when S2 has never heard of it ──

class _ZenodoResp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def _zenodo(monkeypatch, resp, calls=None):
    """Stub the ONE Zenodo request. `calls` collects urls, so a test can prove the request was
    never MADE rather than only that the result was empty."""
    import requests
    def fake_get(url, **kw):
        if calls is not None:
            calls.append(url)
        return resp
    monkeypatch.setattr(requests, "get", fake_get)


_ZENODO_PAPER = {"externalIds": {"DOI": "10.5281/zenodo.21921441"}}
_ZENODO_REC = {
    "metadata": {"title": "Continuous Memory for Multi-Agent Infrastructure",
                 "creators": [{"name": "Nuraliev, Ravshan"}, {"name": "Second, Author"}],
                 "publication_date": "2026-08-13",
                 "description": "<p>An abstract with <strong>markup</strong> in it.</p>"},
    "files": [
        {"key": "ClaimKeep Paper v0.11.pdf",
         "links": {"self": "https://zenodo.org/api/records/21921441/files/paper.pdf/content"}},
        {"key": "source.md",
         "links": {"self": "https://zenodo.org/api/records/21921441/files/source.md/content"}},
    ],
}


def test_zenodo_supplies_the_metadata_s2_could_not(monkeypatch):
    """S2 resolved 1 of 15 OpenAlex DOIs; Zenodo is the biggest group of the 404s. Without this the
    Paper has no title, and every caller passing no `known=` mints `Untitled` — permanently."""
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC))
    got = ip._zenodo_metadata(_ZENODO_PAPER)
    assert got["title"].startswith("Continuous Memory")
    assert [a["name"] for a in got["authors"]] == ["Nuraliev, Ravshan", "Second, Author"]
    assert got["publicationDate"] == "2026-08-13"


def test_the_zenodo_abstract_arrives_as_text_not_markup():
    """Zenodo ships its description as HTML; an abstract must read the same however it entered."""
    out = ip._html_to_text("<p>An abstract with <strong>markup</strong> in it.</p>")
    assert "markup" in out and "<strong>" not in out


def test_the_same_record_carries_the_body_so_no_second_request_is_needed(monkeypatch):
    """The pdf path embeds the DEPOSITOR's filename and cannot be derived from the DOI — but it
    arrives in the SAME record as the title. Handing it over as `openAccessPdf` is what lets
    `_fulltext_pdf_urls` stay pure and offline."""
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC))
    got = ip._zenodo_metadata(_ZENODO_PAPER)
    assert got["openAccessPdf"]["url"].endswith("/paper.pdf/content")   # the .md is ignored
    assert ip._fulltext_pdf_urls({"externalIds": {}, **got})[0] == got["openAccessPdf"]["url"]


def test_a_software_deposit_yields_metadata_but_no_body(monkeypatch):
    """Zenodo hosts code as well as papers. Filtering `files[]` to `.pdf` is what makes a software
    deposit correctly bodyless — there is no `resource_type` branch to maintain. It still gets a
    title, so it mints as an honest abstract-only atom instead of an `Untitled` one."""
    rec = {**_ZENODO_REC, "files": [{"key": "TrustAdaptRL-v1.zip",
                                     "links": {"self": "https://zenodo.org/x.zip"}}]}
    _zenodo(monkeypatch, _ZenodoResp(200, rec))
    got = ip._zenodo_metadata(_ZENODO_PAPER)
    assert got["title"] and "openAccessPdf" not in got


@pytest.mark.parametrize("resp", [_ZenodoResp(404), _ZenodoResp(500), _ZenodoResp(200, {})])
def test_zenodo_failures_leave_the_caller_with_what_it_had(monkeypatch, resp):
    """Fail-safe: a missing record, an outage, or an unexpected shape must all return None so the
    caller keeps its own metadata, never raise into the ingest loop."""
    _zenodo(monkeypatch, resp)
    assert ip._zenodo_metadata(_ZENODO_PAPER) is None


def test_a_non_zenodo_doi_never_spends_a_request(monkeypatch):
    calls = []
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC), calls)
    assert ip._zenodo_metadata({"externalIds": {"DOI": "10.1234/ordinary"}}) is None
    assert ip._zenodo_metadata({"externalIds": {}}) is None
    assert calls == []


def test_the_fallback_fires_only_when_there_is_no_body(monkeypatch):
    """The cost gate. A caller that supplied `known=` (the frontier path) or an S2 that answered
    fully leaves a BODY behind, and neither pays for a request."""
    called = []
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    monkeypatch.setattr(ip, "_openalex_metadata", lambda paper: None)
    monkeypatch.setattr(ip, "_zenodo_metadata", lambda paper: called.append(paper) or None)
    ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True,
                      known={"title": "The finder already knew this",
                             "abstract": "and it already had the abstract too"})
    assert called == []


def test_an_s2_404_on_a_zenodo_doi_no_longer_mints_untitled(monkeypatch):
    """The regression this closes, end to end through `paper_from_url`. Two of three paper callers
    pass no `known=`, and `ingest_x_footprint` mints `author_referenced` atoms — human-attested —
    so the bad row landed in the tier the KB is built on."""
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    _openalex(monkeypatch, _FakeWorksAdapter(None))   # the wider index misses; Zenodo answers
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC))
    paper = ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True)
    assert paper["title"].startswith("Continuous Memory")
    assert paper["abstract"] and paper["authors"]
    assert "Untitled" not in ip.paper_to_markdown_full(paper, None)


# ── OpenAlex metadata fallback: the index that has the paper S2 has not reached yet ──
#
# Measured 2026-09-11: of 7 DOIs S2 could not resolve (days-old ACS/RSC articles), OpenAlex had
# title + abstract for 7. Of 12 DOIs S2 COULD resolve, OpenAlex added a pdf for 0 — which is why
# the gate below matters as much as the fill.

def _inverted(text):
    """A sentence → OpenAlex's `{word: [positions]}` inverted abstract index, the wire shape."""
    idx = {}
    for i, w in enumerate(text.split()):
        idx.setdefault(w, []).append(i)
    return idx


_OA_ABSTRACT = "Viologen cyclophanes show through space conjugation across the macrocycle."
_OA_WORK = {
    "title": "Through-Space-Conjugated Viologen Cyclophanes",
    "abstract_inverted_index": _inverted(_OA_ABSTRACT),
    "primary_location": {"source": {"display_name": "Journal of the American Chemical Society"},
                         "pdf_url": None},
    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2511.23155"},
    "publication_date": "2026-09-02",
    "publication_year": 2026,
    "cited_by_count": 3,
    "authorships": [
        {"author": {"id": "https://openalex.org/A5043841592", "display_name": "Ada Chemist",
                    "orcid": "https://orcid.org/0000-0002-4027-364X"},
         "author_position": "first"},
        {"author": {"id": "https://openalex.org/A5000000002", "display_name": "Bo Synthesist"},
         "author_position": "last"},
    ],
}
_OA_URL = "https://pubs.acs.org/doi/10.1021/jacs.6c13064"
_OA_ATOM = "paper:DOI:10.1021/jacs.6c13064"


def _s2_silent(monkeypatch):
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))


def test_openalex_fills_metadata_when_s2_is_silent(conn, fake_embedder, monkeypatch):
    """The bug, end to end. Before this, the DOI parsed perfectly and every other field was empty,
    so `atomize_paper`'s no-body skip fired and the deposit was lost ENTIRELY — not a thin atom,
    nothing. That skip is correct; the defect was never asking the one index that knew."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK))
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert paper["title"].startswith("Through-Space-Conjugated")
    assert "through space conjugation" in paper["abstract"]
    assert [a["name"] for a in paper["authors"]] == ["Ada Chemist", "Bo Synthesist"]
    assert paper["year"] == 2026 and paper["publicationDate"] == "2026-09-02"
    assert paper["venue"] == "Journal of the American Chemical Society"
    # `openalexId`, never `authorId`: `derive_paper` mints `who_id = scholar:{id}` off `authorId`,
    # and an OpenAlex author id is not a Semantic Scholar one.
    assert paper["authors"][0]["openalexId"] == "A5043841592"
    assert "authorId" not in paper["authors"][0]
    assert paper["authors"][0]["orcid"] == "0000-0002-4027-364X"

    _stub_fulltext(monkeypatch, None)          # abstract-only, the case that used to SKIP
    assert ip.atomize_paper(conn, fake_embedder, paper) == _OA_ATOM


def test_openalex_is_not_called_when_s2_answered(monkeypatch):
    """The cost gate. OpenAlex added a pdf S2 lacked in 0 of 12 papers S2 resolved, so a paper that
    already has a body must never spend the request."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"title": "S2 already knows this", "authors": [],
                                         "abstract": "and S2 supplied the abstract as well."},
                                        ip.FETCH_OK))
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources, "OpenAlexWorksAdapter",
                        lambda *a, **kw: pytest.fail("OpenAlex must not be asked"))
    assert ip.paper_from_url(_OA_URL, enrich=True)["title"] == "S2 already knows this"


def test_a_title_with_no_abstract_still_asks_openalex(monkeypatch):
    """The gate is a missing BODY, not a missing title — the two are different questions, and the
    difference is 9 real papers. S2 answers for plenty of older work with a title and nothing else;
    `atomize_paper` then drops it for having no body, and before 2026-09-11 the resolver that could
    have supplied one was skipped precisely BECAUSE a title was there. Measured over 129 pasted
    urls: 12 landed in this state, OpenAlex held an abstract for 9 (LeCun's gradient-based
    learning paper, Tibshirani's lasso, the reproducibility-project paper)."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"title": "Gradient-based learning applied to documents",
                                         "authors": [{"authorId": "111", "name": "Y. LeCun"}]},
                                        ip.FETCH_OK))
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK))
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert "through space conjugation" in paper["abstract"]


def test_filling_a_gap_never_overwrites_what_s2_answered(monkeypatch):
    """`_fill_gaps`, not `paper.update` — and this is the reason. A paper reaching the widened gate
    usually HAS authors from S2, carrying `authorId`, which is what mints `who_id = scholar:{id}`.
    Overwriting them with OpenAlex's `openalexId` authors would silently cost every such paper its
    author identity, and nothing downstream would report it."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"title": "The title S2 already answered with",
                                         "authors": [{"authorId": "111", "name": "Alice S2"}],
                                         "venue": "NeurIPS", "year": 2017},
                                        ip.FETCH_OK))
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK))
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert paper["title"] == "The title S2 already answered with"
    assert [a["name"] for a in paper["authors"]] == ["Alice S2"]
    assert paper["authors"][0]["authorId"] == "111"     # `who_id = scholar:111` survives
    assert paper["venue"] == "NeurIPS" and paper["year"] == 2017
    assert paper["abstract"]                            # the actual GAP was filled


def test_openalex_never_moves_the_atom_id(monkeypatch):
    """METADATA ONLY. OpenAlex will report that a JACS paper also exists on arXiv; writing that id
    would re-key the atom off the url the user actually pasted, and Policy B then freezes the
    wrong one forever. Identity belongs to `_parse_paper_url`."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter({**_OA_WORK, "id": "https://openalex.org/W123",
                                              "doi": "https://doi.org/10.48550/arxiv.2511.23155",
                                              "ids": {"arxiv": "2511.23155"}}))
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert ip.paper_atom_id(paper) == _OA_ATOM
    assert "ArXiv" not in paper["externalIds"] and paper["url"] == _OA_URL
    assert paper["paperId"] == "DOI:10.1021/jacs.6c13064"


def test_openalex_pdf_reaches_fulltext_urls(monkeypatch):
    """`best_oa_location` is the authors' own preprint of a paywalled paper — so this recovers part
    of the paywall case without going near a paywall. `_fulltext_pdf_urls` already reads
    `openAccessPdf`, so it needed no edit and the on-open deepen gets the body for free."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK))
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert ip._fulltext_pdf_urls(paper) == ["https://arxiv.org/pdf/2511.23155"]


@pytest.mark.parametrize("adapter", [
    _FakeWorksAdapter(_OA_WORK, up=False),                       # breaker open — no request at all
    _FakeWorksAdapter(raises=RuntimeError("breaker open — backing off")),
    _FakeWorksAdapter(None),                                     # indexed nowhere: count=0
    _FakeWorksAdapter("not a dict"),                             # a shape nobody planned for
])
def test_openalex_failure_degrades_to_today(monkeypatch, adapter):
    """LOAD-BEARING fail-safe: `paper_from_url` is called from hopper, `link_router.mint_artifact`
    and the X prefetch, and may never raise. Every failure returns the un-enriched paper — exactly
    the behaviour we already had."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, adapter)
    paper = ip.paper_from_url(_OA_URL, enrich=True)
    assert paper["paperId"] == "DOI:10.1021/jacs.6c13064"
    assert not paper.get("title")


def test_the_breaker_being_open_costs_no_request(monkeypatch):
    """`available()` FIRST, the way `oracles._openalex_root` does it: a host already known to be
    down must not cost a request to rediscover that."""
    calls = []
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK, up=False, calls=calls))
    ip.paper_from_url(_OA_URL, enrich=True)
    assert calls == []


def test_openalex_empties_never_clobber(monkeypatch):
    """The caller merges with `paper.update()`, so a null field in the record must not ERASE what
    a `known=` caller already supplied. Empties are stripped before the merge."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter({**_OA_WORK, "abstract_inverted_index": None,
                                              "best_oa_location": None}))
    paper = ip.paper_from_url(_OA_URL, enrich=True,
                              known={"abstract": "The finder already had this abstract."})
    assert paper["abstract"] == "The finder already had this abstract."
    assert paper["title"].startswith("Through-Space-Conjugated")   # the fill still happened
    assert not paper.get("openAccessPdf")


def test_zenodo_still_runs_when_openalex_misses(monkeypatch):
    """The chain order. OpenAlex resolved a Zenodo DOI in testing and probably subsumes the Zenodo
    call — but "probably" is not the standard this repo deletes a working path on."""
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(None))
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC))
    paper = ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True)
    assert paper["title"].startswith("Continuous Memory")


def test_openalex_wins_over_zenodo_when_both_could_answer(monkeypatch):
    """OpenAlex is asked first: it is the wider index (~250M works against one repository), and
    whichever fills the title stops the chain, so Zenodo never spends its request."""
    calls = []
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(_OA_WORK))
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC), calls)
    paper = ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True)
    assert paper["title"].startswith("Through-Space-Conjugated")
    assert calls == []


# ── the paced door onto OpenAlex ─────────────────────────────────────────────────

def test_reads_are_paced_to_the_adapters_declared_interval(monkeypatch):
    """`min_interval_s` is DECLARED on the adapter and APPLIED BY THE CALLER, and the paper path
    was never one of the callers that applied it — it built a fresh adapter per read and fired at
    once. Harmless while the only read was a rare missing-title fallback; not harmless once three
    reads landed here. A 293-url run on 2026-09-11 took OpenAlex to HTTP 429, opened the PERSISTED
    breaker, and every read returned None for 15 minutes — including the metadata resolver that
    had been working. The stored-atom rate fell 56% -> 47% and nothing said why, because a
    breaker-open read looks exactly like "OpenAlex has never heard of this paper".

    Clock and sleep are both faked: this asserts the SPACING, not that the suite waited."""
    from pipeline.kb import frontier_sources
    now, slept = [1000.0], []
    monkeypatch.setattr(ip.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(ip.time, "sleep", lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "min_interval_s", 2.0)
    monkeypatch.setattr(ip, "_OPENALEX_NEXT_AT", 0.0)
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "available", lambda self: True)

    for _ in range(3):
        ip._openalex_read(lambda a: {"ok": True})
    assert slept == [2.0, 2.0], "the first read goes at once; each next one waits the interval"


def test_a_read_never_raises_and_never_paces_a_dead_host(monkeypatch):
    """`available()` is checked BEFORE the delay — a host known to be down must not also cost a
    second per caller to rediscover that."""
    from pipeline.kb import frontier_sources
    slept = []
    monkeypatch.setattr(ip.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "available", lambda self: False)
    assert ip._openalex_read(lambda a: pytest.fail("breaker is open")) is None
    assert slept == []
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "available", lambda self: True)
    assert ip._openalex_read(lambda a: (_ for _ in ()).throw(RuntimeError("429"))) is None


# ── three more front doors onto ids we already resolve ───────────────────────────

def test_the_corpus_id_form_of_a_semantic_scholar_url(monkeypatch):
    """semanticscholar.org was already a paper host, but `_S2_RE` matched only a 40-hex hash or
    bare digits — so its `CorpusID:` form routed to the paper adapter and then refused."""
    p = ip.paper_from_url("https://www.semanticscholar.org/paper/CorpusID:13756489", enrich=False)
    assert ip.paper_atom_id(p) == "paper:CorpusID:13756489"


def test_europe_pmc_reaches_the_same_atom_as_pubmed(monkeypatch):
    """Europe PMC is a second front door onto ids we already resolve — no new lookup, just the
    two E-utilities calls the PubMed and PMC branches already make."""
    _stub_eutils(monkeypatch, "10.1038/s41586-021-03819-2")
    _openalex(monkeypatch, _FakeWorksAdapter(None))
    p = ip.paper_from_url("https://europepmc.org/article/MED/34265844", enrich=True)
    assert ip.paper_atom_id(p) == "paper:DOI:10.1038/s41586-021-03819-2"
    assert p["url"] == "https://europepmc.org/article/MED/34265844"


@pytest.mark.parametrize("url,db,uid", [
    ("https://europepmc.org/article/PMC/PMC8371605", "pmc", "8371605"),
    ("https://europepmc.org/abstract/PMC/PMC8371605", "pmc", "8371605"),
    ("https://europepmc.org/articles/PMC8371605", "pmc", "8371605"),   # their legacy form
])
def test_europe_pmcs_own_pmc_view_resolves(monkeypatch, url, db, uid):
    """Europe PMC addresses a PMC copy as `/article/PMC/PMC…` — the SOURCE segment, then the id.
    The pattern wanted `articles?/PMC` until 2026-09-16 and so matched only the legacy form, which
    meant their own PMC url fell through to the BLOG ingester. Invisible in both directions: the
    host answers 403 to the `citation_doi` probe, so nothing ever contradicted the mis-route."""
    seen = {}
    monkeypatch.setattr(ip, "_eutils_doi",
                        lambda d, u: seen.update(db=d, uid=u) or "10.1038/s41586-021-03819-2")
    assert ip._looked_up_doi(url) == "10.1038/s41586-021-03819-2"
    assert seen == {"db": db, "uid": uid}


def test_a_europe_pmc_preprint_asks_europe_pmc_not_ncbi(monkeypatch):
    """PPR is a PREPRINT id, and NCBI has never heard of it — E-utilities answers for PubMed and
    PMC only. Europe PMC indexes them itself and returns the DOI the preprint server minted."""
    calls = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"resultList": {"result": [{"id": "PPR217527", "source": "PPR",
                                               "doi": "10.21203/rs.3.rs-76053/v1"}]}}

    monkeypatch.setattr(ip.requests, "get",
                        lambda url, **kw: (calls.update(url=url, **kw), _Resp())[1])
    monkeypatch.setattr(ip, "_eutils_doi", lambda *a: pytest.fail("NCBI cannot answer for a PPR"))
    assert ip._looked_up_doi("https://europepmc.org/article/PPR/PPR217527") == (
        "10.21203/rs.3.rs-76053/v1")
    # SRC:PPR is part of the query, not decoration: a Europe PMC id is unique only WITHIN a
    # source, so an unqualified EXT_ID search can answer about a different database's record.
    assert calls["params"]["query"] == "EXT_ID:PPR217527 AND SRC:PPR"


def test_every_europe_pmc_lookup_is_fail_safe(monkeypatch):
    """The invariant each of these shares: a bad answer is None and the url stays unparsed — never
    a raise, and never a guess at the id."""
    class _Bad:
        status_code = 500
        @staticmethod
        def json():
            raise ValueError("not json")

    monkeypatch.setattr(ip.requests, "get", lambda *a, **kw: _Bad())
    assert ip._europepmc_ppr_doi("PPR217527") is None


def test_an_openalex_work_page_resolves_to_the_paper(monkeypatch):
    _s2_silent(monkeypatch)
    _openalex(monkeypatch, _FakeWorksAdapter(
        None, by_id={"doi": "https://doi.org/10.7717/peerj.4375"}))
    p = ip.paper_from_url("https://openalex.org/W2741809807", enrich=True)
    assert ip.paper_atom_id(p) == "paper:DOI:10.7717/peerj.4375"


def test_an_openalex_work_lookup_failure_refuses(monkeypatch):
    _openalex(monkeypatch, _FakeWorksAdapter(None, raises=RuntimeError("down")))
    assert ip.paper_from_url("https://openalex.org/W2741809807", enrich=True) is None


# ── the DOI a url PRINTS vs the url decoration wrapped around it ──────────────────
#
# A DOI may contain slashes (`10.1088/1748-9326/ab4553`), so the capture cannot stop at the first
# one — and therefore also swallows whatever view segment the publisher appended. 5 of 85 parsed
# DOIs were wrong this way (2026-09-11), and every wrong one resolved to nothing in OpenAlex.

@pytest.mark.parametrize("url,doi", [
    ("https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full", "10.3389/fpsyg.2013.00863"),
    ("https://www.degruyter.com/document/doi/10.1515/9783110769043-005/html", "10.1515/9783110769043-005"),
    ("https://www.numdam.org/articles/10.1007/s10240-012-0042-x/", "10.1007/s10240-012-0042-x"),
    ("https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1665593", "10.1080/00401706.2019.1665593"),
    ("https://example.org/doi/10.1234/abcd/references", "10.1234/abcd"),
])
def test_a_view_segment_is_not_part_of_the_doi(url, doi):
    assert ip.paper_from_url(url, enrich=False)["externalIds"]["DOI"] == doi


@pytest.mark.parametrize("url", [
    "https://www.biorxiv.org/content/10.1101/2020.03.22.002386v1",
    "https://www.biorxiv.org/content/10.1101/2020.03.22.002386v1.full",
    "https://www.biorxiv.org/content/10.1101/2020.03.22.002386v2.full.pdf",
    "https://www.biorxiv.org/content/10.1101/2020.03.22.002386",
])
def test_biorxiv_versions_all_mint_one_atom(url):
    """The serious case. bioRxiv and medRxiv are first-class paper hosts and THIS is their
    canonical url, so the store was minting `paper:DOI:…002386v1` — an id that resolves to nothing
    and, being immutable, would never dedup against the same preprint pasted as a clean DOI. The
    version lives in the url, never in the DOI — exactly arXiv's situation one prefix over."""
    assert ip.paper_atom_id(ip.paper_from_url(url, enrich=False)) == \
        "paper:DOI:10.1101/2020.03.22.002386"


@pytest.mark.parametrize("url,doi", [
    # a DOI that genuinely ends in something unusual must survive untouched
    ("https://doi.org/10.1002/1521-3773(20010601)40:11<2004::aid-anie2004>3.0.co;2-5",
     "10.1002/1521-3773(20010601)40:11<2004::aid-anie2004>3.0.co;2-5"),
    ("https://iopscience.iop.org/article/10.1088/1748-9326/ab4553", "10.1088/1748-9326/ab4553"),
    ("https://onlinelibrary.wiley.com/doi/pdfdirect/10.3322/caac.21834", "10.3322/caac.21834"),
    ("https://link.springer.com/article/10.1007/s11263-015-0816-y", "10.1007/s11263-015-0816-y"),
])
def test_the_trim_only_removes_what_it_recognizes(url, doi):
    """Conservative by construction: only a KNOWN view word is ever removed. A slash-bearing DOI
    and a `/pdfdirect/` PREFIX (which is not a tail at all) both come through intact."""
    assert ip.paper_from_url(url, enrich=False)["externalIds"]["DOI"] == doi


# ── _openalex_doi_by_url: the last-resort identity, and the judgement it refuses to make ──

def test_a_url_only_the_index_can_read_resolves_to_its_paper(monkeypatch):
    """`repositorio.unal.edu.co/handle/unal/81443` is ResNet. 59 of 96 refused urls are this
    shape: an ordinary paper wearing a url we cannot read, on a host that usually will not even
    serve us the page."""
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "works_by_landing_page",
                        lambda self, url: [{"doi": "https://doi.org/10.1109/CVPR.2016.90",
                                            "title": "Deep Residual Learning"}])
    assert ip._openalex_doi_by_url("https://repositorio.unal.edu.co/handle/unal/81443") == \
        "10.1109/cvpr.2016.90"


def test_two_papers_claiming_one_page_is_refused_not_guessed(monkeypatch):
    """The same uniqueness rule `oracles._openalex_root` applies to venue names. More than one
    work claiming a page means we cannot say which paper the user meant, and a wrong paper atom is
    immutable. Measured: 1 of 33 recovered pages was ambiguous — rare, and therefore easy to get
    wrong by not looking."""
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "works_by_landing_page",
                        lambda self, url: [{"doi": "https://doi.org/10.1/a"},
                                           {"doi": "https://doi.org/10.2/b"}])
    assert ip._openalex_doi_by_url("https://repo.example/handle/1") is None


@pytest.mark.parametrize("works", [[], [{"title": "no doi at all"}], [{"doi": ""}]])
def test_the_last_resort_refuses_rather_than_inventing(monkeypatch, works):
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "works_by_landing_page",
                        lambda self, url: works)
    assert ip._openalex_doi_by_url("https://repo.example/handle/1") is None


def test_the_last_resort_is_fail_safe(monkeypatch):
    """It runs inside `classify_link_deep`, which may never raise."""
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "works_by_landing_page",
                        lambda self, url: (_ for _ in ()).throw(RuntimeError("breaker open")))
    assert ip._openalex_doi_by_url("https://repo.example/handle/1") is None
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "available", lambda self: False)
    assert ip._openalex_doi_by_url("https://repo.example/handle/1") is None


def test_the_last_resort_never_runs_on_a_non_url(monkeypatch):
    from pipeline.kb import frontier_sources
    monkeypatch.setattr(frontier_sources.OpenAlexWorksAdapter, "works_by_landing_page",
                        lambda self, url: pytest.fail("not a url — nothing to look up"))
    assert ip._openalex_doi_by_url("10.1038/s41586-021-03819-2") is None
    assert ip._openalex_doi_by_url("") is None


# ── the url forms a reader actually pastes (added 2026-09-11 from a 129-url sweep) ──
#
# Every case here was REFUSED before that sweep, and each was refused for a different reason:
# a host that serves papers under one path only, an id that is a DOI in disguise, an id that needs
# one request, and a rendered view arXiv now serves by default.

@pytest.mark.parametrize("url,atom", [
    ("https://arxiv.org/html/2402.17764v1", "paper:arXiv:2402.17764"),
    ("https://arxiv.org/html/2402.17764v1#S3.T2", "paper:arXiv:2402.17764"),
    ("https://huggingface.co/papers/2005.14165", "paper:arXiv:2005.14165"),
    ("https://www.alphaxiv.org/abs/2005.14165", "paper:arXiv:2005.14165"),
    ("https://www.alphaxiv.org/overview/2005.14165v4", "paper:arXiv:2005.14165"),
])
def test_an_arxiv_id_mints_one_atom_whatever_front_end_shows_it(url, atom):
    """arXiv's own HTML view — now the DEFAULT for recent papers — plus the two reader mirrors.
    All of them are the same preprint, so all of them must be the same atom; a second key would
    split an immutable paper by which front-end the reader happened to be looking at."""
    assert ip.paper_atom_id(ip.paper_from_url(url, enrich=False)) == atom


@pytest.mark.parametrize("url", [
    "https://huggingface.co/meta-llama/Llama-3-8B",
    "https://huggingface.co/datasets/squad",
    "https://www.alphaxiv.org/",
])
def test_the_mirror_branches_do_not_swallow_the_rest_of_the_host(url):
    """Hugging Face is mostly models and datasets. The id pattern is restricted to arXiv's modern
    `YYMM.NNNNN` form precisely so a model page cannot become a paper id — the loose `.+?` that
    arXiv's own branch needs for legacy `hep-th/9901001` ids would do exactly that here."""
    assert ip.paper_from_url(url, enrich=False) is None


@pytest.mark.parametrize("url", [
    "https://ssrn.com/abstract=3482150",
    "https://www.ssrn.com/abstract=3482150",
    "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3482150",
    "https://papers.ssrn.com/sol3/papers.cfm?foo=bar&abstract_id=3482150",
])
def test_ssrn_resolves_offline_because_its_id_is_a_doi_in_disguise(url):
    """SSRN mints `10.2139/ssrn.{abstract_id}`, so no request is needed — which matters because
    ssrn.com answers 403 to a server fetch (browser User-Agent included), so the `citation_doi`
    probe can never reach it. Checked before trusting: 10 of 10 sampled SSRN works carry that
    prefix and 3 of 3 ids round-tripped through Crossref (2026-09-11)."""
    p = ip.paper_from_url(url, enrich=False)
    assert ip.paper_atom_id(p) == "paper:DOI:10.2139/ssrn.3482150"
    assert p["url"] == url                      # the user's own url is what gets stored
    # and it is the SAME atom as the DOI form of that paper
    assert ip.paper_atom_id(ip.paper_from_url("https://doi.org/10.2139/ssrn.3482150",
                                              enrich=False)) == ip.paper_atom_id(p)


def test_ssrn_stays_offline(monkeypatch):
    """`predicted_atom_id` rests on `_parse_paper_url` being pure string work. SSRN must not spend
    a request the way PubMed has to."""
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **kw: pytest.fail("SSRN needs no lookup"))
    assert ip.paper_from_url("https://ssrn.com/abstract=3482150", enrich=False) is not None


def _stub_pmc(monkeypatch, doi, uid="8371605"):
    calls = []

    def _get(url, **kw):
        calls.append((url, (kw.get("params") or {}).get("db")))
        ids = [{"idtype": "pmcid", "value": f"PMC{uid}"}]
        if doi:
            ids.append({"idtype": "doi", "value": doi})
        return _Resp(200, {"result": {uid: {"articleids": ids}}})
    import requests
    monkeypatch.setattr(requests, "get", _get)
    return calls


@pytest.mark.parametrize("url", [
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8371605/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/8371605",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/",
])
def test_a_pmc_url_mints_the_same_atom_as_its_doi(monkeypatch, url):
    """PMC is the largest open-access full-text archive in biomedicine and was refused outright —
    and unlike nature.com it cannot be rescued by the page probe, because ncbi answers 403 to a
    server fetch. The lookup is the same E-utilities call a PMID already makes, with `db=pmc`."""
    calls = _stub_pmc(monkeypatch, "10.1038/s41586-021-03819-2")
    _openalex(monkeypatch, _FakeWorksAdapter(None))
    p = ip.paper_from_url(url, enrich=True)
    assert ip.paper_atom_id(p) == "paper:DOI:10.1038/s41586-021-03819-2"
    assert p["url"] == url
    assert calls and calls[0][1] == "pmc"       # asked the PMC database, not pubmed


def test_a_pmc_record_with_no_doi_refuses_rather_than_guessing(monkeypatch):
    """Fail-safe, landing on the old behaviour rather than an invented id."""
    _stub_pmc(monkeypatch, None)
    assert ip.paper_from_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/",
                             enrich=True) is None


def test_pmc_stays_offline_when_enrich_is_false(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("no network on this path"))
    assert ip.paper_from_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/",
                             enrich=False) is None


def test_a_zenodo_record_page_uses_the_doi_the_record_declares(monkeypatch):
    """NOT `10.5281/zenodo.{record_id}`, which looks free and is wrong: a record number can be a
    CONCEPT id standing for all versions, whose current version carries a DIFFERENT DOI. Measured
    2026-09-11: `records/20027463` declares `…20027464`, and `records/3509134` declares
    `…21500199`. Deriving would give the deposit a second permanent atom id."""
    import requests
    seen = []

    def _get(url, **kw):
        seen.append(url)
        return _ZenodoResp(200, {"doi": "10.5281/zenodo.20027464",
                                 "metadata": {"title": "Lista de Verificação"}})
    monkeypatch.setattr(requests, "get", _get)
    _openalex(monkeypatch, _FakeWorksAdapter(None))
    p = ip.paper_from_url("https://zenodo.org/records/20027463", enrich=True)
    assert ip.paper_atom_id(p) == "paper:DOI:10.5281/zenodo.20027464"   # NOT …463
    assert any("/api/records/20027463" in u for u in seen)


def test_a_zenodo_record_lookup_failure_degrades_to_refused(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("zenodo down")))
    assert ip.paper_from_url("https://zenodo.org/records/20027463", enrich=True) is None


# ── the PDF's own /Title: the last metadata source, and why it is precision-first ──

def _pdf_with_title(title):
    """A REAL one-page pdf carrying `/Title` — no mocked reader, so these exercise pypdf itself."""
    import io
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(200, 200)
    if title is not None:
        w.add_metadata({"/Title": title})
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_a_real_pdf_title_is_read():
    assert ip._pdf_title(_pdf_with_title("Capable but Not Deployable: Institutional Constraints")) \
        == "Capable but Not Deployable: Institutional Constraints"


@pytest.mark.parametrize("junk", [
    "Microsoft Word - draft3.docx",      # the authoring tool's default
    "cesifo_wp_final.pdf",               # a filename nobody replaced
    "IHR-06-2025-0057_proof 1..26",      # a typesetter proof stamp
    "0123456789",                        # an accession id with no words
    "untitled",
    "Report",                            # too short to be a title
])
def test_junk_title_values_are_rejected_rather_than_written(junk):
    """A WRONG title is worse than none: policy B freezes it, and it reads as true forever. Same
    principle as the FETCH_ABSENT fix — an honest gap beats a false claim."""
    assert ip._pdf_title(_pdf_with_title(junk)) is None


def test_a_pdf_with_no_title_metadata_yields_none():
    assert ip._pdf_title(_pdf_with_title(None)) is None


def test_a_corrupt_pdf_degrades_rather_than_raising():
    assert ip._pdf_title(b"<!DOCTYPE html><html>not a pdf</html>") is None


def test_resolve_fulltext_fills_a_missing_title_from_the_pdf(monkeypatch):
    """The regression this closes. A raw hosted `.pdf` link gets `s2_lookup: None`, so S2 is never
    even called, and `openAccessPdf` is the url itself — so the body always resolves and the title
    is always absent. Every such link through the hopper or an Oracle's footprint minted
    `# Untitled` with a full body, permanently."""
    paper = ip.paper_from_url("https://repo.example.edu/docs/wp12941.pdf", enrich=False)
    assert not paper.get("title")                       # nothing offline can supply one
    monkeypatch.setattr(ip, "_download_pdf", lambda url: _pdf_with_title("The Actual Paper Title"))
    monkeypatch.setattr(ip, "_pdf_bytes_to_text", lambda data: "a substantive body")
    assert ip.resolve_fulltext(paper) == "a substantive body"
    assert paper["title"] == "The Actual Paper Title"
    assert "Untitled" not in ip.paper_to_markdown_full(paper, "a substantive body")


def test_a_title_that_is_already_known_is_never_overwritten(monkeypatch):
    """Precedence: S2, `known=` and `_zenodo_metadata` all win. The pdf is the LAST source, used
    only where nothing better exists."""
    paper = ip.paper_from_url("https://repo.example.edu/docs/wp12941.pdf", enrich=False,
                              known={"title": "What the finder already knew"})
    monkeypatch.setattr(ip, "_download_pdf", lambda url: _pdf_with_title("A Worse Pdf Title"))
    monkeypatch.setattr(ip, "_pdf_bytes_to_text", lambda data: "a substantive body")
    ip.resolve_fulltext(paper)
    assert paper["title"] == "What the finder already knew"


def test_no_usable_pdf_title_leaves_the_paper_honestly_untitled(monkeypatch):
    """Coverage is 63% by design and the rest keep no title. Never invent one to fill the gap."""
    paper = ip.paper_from_url("https://repo.example.edu/docs/wp12941.pdf", enrich=False)
    monkeypatch.setattr(ip, "_download_pdf", lambda url: _pdf_with_title("Microsoft Word - x.docx"))
    monkeypatch.setattr(ip, "_pdf_bytes_to_text", lambda data: "a substantive body")
    ip.resolve_fulltext(paper)
    assert not paper.get("title")


# ── extraction fidelity: the completeness floors, and the word-level repairs ──────

def _stub_extractor(monkeypatch, text, pages):
    """Replace the pypdf MECHANICS so the policy in `_pdf_bytes_to_text` can be tested directly."""
    monkeypatch.setattr(ip, "_extract_pypdf", lambda data: (text, pages))


def test_a_long_scan_that_clears_the_absolute_floor_is_still_rejected(monkeypatch):
    """The bug the per-page floor exists for. A digitized PDF whose only text layer is a per-page
    copyright watermark scales its char count with page count, so a long one clears an ABSOLUTE
    floor while missing the entire document — and would then be stamped `body_state: complete`,
    permanently, under policy-B dedup. Shape and rate are measured, not invented: Nature's
    pre-1930 archive yields exactly '© 1898 Nature Publishing Group' per page, ~30 chars."""
    watermark = "© 1898 Nature Publishing Group\n\n" * 20      # 20 pages, ~620 chars
    assert len(watermark) >= ip._MIN_FULLTEXT_CHARS              # clears the absolute floor…
    _stub_extractor(monkeypatch, watermark, 20)
    assert ip._pdf_bytes_to_text(b"%PDF") is None                # …and is still rejected.


def test_the_same_watermark_on_one_page_is_rejected_by_the_absolute_floor(monkeypatch):
    """The other half of why BOTH floors are kept. At `_FLOOR_PER_PAGE` a single page needs only
    200 chars, which is WEAKER than the absolute floor — so the absolute floor is what catches a
    short stub, and the per-page floor is what catches a long one."""
    _stub_extractor(monkeypatch, "© 1921 Nature Publishing Group", 1)
    assert ip._pdf_bytes_to_text(b"%PDF") is None


def test_a_genuine_document_passes_both_floors(monkeypatch):
    """Measured over 505 live PDFs, no genuine document fell below 787 chars/page; the floor sits
    at 200. A real body must clear both floors comfortably or the guard is a false-positive
    machine."""
    body = "This is a real paragraph of a real paper. " * 60    # ~2,500 chars/page over 1 page
    _stub_extractor(monkeypatch, body * 20, 20)
    assert ip._pdf_bytes_to_text(b"%PDF") is not None


def test_extractor_failure_degrades_rather_than_raising(monkeypatch):
    """LOAD-BEARING fail-safe: a corrupt PDF must return None so the caller writes an honest
    abstract-only atom, never propagate an exception into the ingest loop."""
    def boom(data):
        raise ValueError("invalid pdf header")
    monkeypatch.setattr(ip, "_extract_pypdf", boom)
    assert ip._pdf_bytes_to_text(b"<!DOCTYPE html>") is None


def test_ligatures_are_folded_so_the_word_is_searchable():
    """A PDF font encodes `fi` as ONE codepoint. Neither BM25 nor the tokenizer matches it against
    "fi", so the word is unreachable by search until it is folded."""
    assert "identified" in ip._repair_extraction("we identi\ufb01ed the cause")
    assert "workflow" in ip._repair_extraction("the work\ufb02ow ran")


def test_a_word_split_across_a_line_break_is_rejoined():
    assert "segmentation" in ip._repair_extraction("we ran seg-\nmentation on it")


def test_a_hyphen_that_belongs_survives_the_rejoin():
    """The rejoin is lowercase-to-lowercase on purpose. Widening it to `\\w` catches 8% more
    line-break hyphens, and every one of those is a hyphen the term genuinely has."""
    for keep in ("GPT-\n4", "ERC-\n8004", "COVID-\n19", "Transformer-\nBERT"):
        assert "-" in ip._repair_extraction(keep)


def test_repair_runs_before_the_floors_so_the_stored_text_is_the_measured_text(monkeypatch):
    """The floors judge what will actually be chunked and embedded, not the raw extraction."""
    _stub_extractor(monkeypatch, "identi\ufb01ed seg-\nmentation. " * 40, 1)
    out = ip._pdf_bytes_to_text(b"%PDF")
    assert "identified" in out and "segmentation" in out
    assert "\ufb01" not in out


# ── The arXiv DOI, and the metadata a caller already holds ────────────────────────
def test_an_arxiv_doi_dedups_onto_the_arxiv_atom_not_a_second_one():
    """arXiv mints a DOI for every preprint, so the same paper reached by DOI and by /abs/ was two
    atoms. Collapsed in `_parse_paper_url` and NOT in any adapter, because this function's stated
    contract is that every link form of one paper reaches one atom — OpenAlex, Semantic Scholar
    and Crossref all hand back this form, so fixing it in one of them re-splits it for the next.
    Checked against the live store before writing (30 paper atoms, none DOI-keyed): no existing
    atom changes identity, which matters because papers are immutable under Policy B."""
    doi = ip.paper_from_url("https://doi.org/10.48550/arxiv.2608.09055", enrich=False)
    abs_ = ip.paper_from_url("https://arxiv.org/abs/2608.09055", enrich=False)
    assert ip.paper_atom_id(doi) == ip.paper_atom_id(abs_) == "paper:arXiv:2608.09055"
    assert doi["externalIds"] == {"ArXiv": "2608.09055"}      # so the OA PDF mirror still resolves
    # And the enrichment lookup follows the fold: S2 knows arXiv ids and 404s the arXiv DOI form.
    assert ip._parse_paper_url("https://doi.org/10.48550/arxiv.2608.09055")["s2_lookup"] == \
        "arXiv:2608.09055"


def test_a_version_suffix_on_an_arxiv_doi_is_stripped_like_any_other():
    """v1 and v2 are the same artifact. The fold has to strip the version too, or the DOI route
    re-introduces exactly the duplicate the /abs/ route already handles."""
    p = ip.paper_from_url("https://doi.org/10.48550/arXiv.2608.09055v3", enrich=False)
    assert ip.paper_atom_id(p) == "paper:arXiv:2608.09055"


def test_an_ordinary_doi_is_untouched_by_the_fold():
    """Only the 10.48550 prefix is arXiv's. Folding any wider would collapse unrelated papers."""
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927", enrich=False)
    assert ip.paper_atom_id(p) == "paper:DOI:10.5281/zenodo.20719927"


def test_known_metadata_survives_when_s2_has_never_heard_of_the_paper(monkeypatch):
    """Measured 2026-08-26: S2 resolved 1 of 15 OpenAlex DOIs. A 404 is `FETCH_ABSENT`, which
    `atomize_paper` does not skip — so without this the finder's own title and abstract would be
    discarded and a contentless atom frozen in permanently."""
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927",
                          known={"title": "Earned Trust", "abstract": "A study."})
    assert (p["title"], p["abstract"]) == ("Earned Trust", "A study.")


def test_a_null_from_s2_never_erases_metadata_the_caller_already_had(monkeypatch):
    """S2 routinely answers with `abstract: null`. Its answer wins where it HAS one; a null is not
    an answer, and letting it overwrite is the same contentless atom by a quieter route."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"title": "S2's better title", "abstract": None,
                                         "venue": "NeurIPS"}, ip.FETCH_OK))
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927",
                          known={"title": "Thin title", "abstract": "The finder's abstract."})
    assert p["title"] == "S2's better title"           # S2 wins where it answered
    assert p["abstract"] == "The finder's abstract."   # and never where it did not
    assert p["venue"] == "NeurIPS"


def test_a_supplied_abstract_stops_a_throttled_s2_fetch_from_skipping(conn, fake_embedder,
                                                                      monkeypatch):
    """The skip's reason is that the atom would be written CONTENTLESS, so a paper carrying its
    own abstract is not the case it guards. Unauthenticated S2 allows ~1 req/s and 429s routinely
    (3 of 3 arXiv lookups, measured 2026-08-26) — skipping these would burn the attempt cap and
    reject real papers. Callers that pass no `known` are unaffected: they have no abstract at this
    point in exactly the cases they did before, which the test above still pins."""
    _stub_fulltext(monkeypatch, None)
    throttled = {**_PAPER, "abstract": "The finder's abstract.",
                 ip._S2_VERDICT: ip.FETCH_UNDETERMINED}

    assert ip.atomize_paper(conn, fake_embedder, throttled) == _ATOM


def test_a_finders_open_pdf_reaches_the_fulltext_resolver(monkeypatch):
    """The full-text seam, end to end through `known=`.

    `_fulltext_pdf_urls` reads `openAccessPdf.url` and always has; the only new thing is WHERE
    that url can come from. S2 does not index Zenodo or most institutional repositories, so for
    those works the finder's own url is the only route to a body — measured 2026-08-26, 25 of 39
    non-arXiv OpenAlex results carry a PDF nothing else in this list can reach.
    """
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927",
                          known={"title": "Earned Trust", "abstract": "A study.",
                                 "openAccessPdf": {"url": "https://example.test/open.pdf"}})
    assert ip._fulltext_pdf_urls(p) == ["https://example.test/open.pdf"]


def test_s2s_own_pdf_still_wins_when_s2_answers_with_one(monkeypatch):
    """`known` is the FALLBACK, not an override. Same rule as title and abstract: S2 wins where it
    answers, and the finder's value survives only where it does not."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"openAccessPdf": {"url": "https://s2.test/paper.pdf"}},
                                        ip.FETCH_OK))
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927",
                          known={"title": "T", "abstract": "A",
                                 "openAccessPdf": {"url": "https://example.test/open.pdf"}})
    assert ip._fulltext_pdf_urls(p) == ["https://s2.test/paper.pdf"]


def test_a_null_openaccesspdf_from_s2_does_not_erase_the_finders(monkeypatch):
    """The nested-check case. The generic keep-ours loop tests `not out.get(k)`, which is False
    for a truthy dict — `{"url": None}` is truthy at the top level and empty inside, so only the
    nested check sees it. Without that, a resolved PDF is lost to an S2 field that says nothing."""
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: ({"openAccessPdf": {"url": None}}, ip.FETCH_OK))
    p = ip.paper_from_url("https://doi.org/10.5281/zenodo.20719927",
                          known={"title": "T", "abstract": "A",
                                 "openAccessPdf": {"url": "https://example.test/open.pdf"}})
    assert ip._fulltext_pdf_urls(p) == ["https://example.test/open.pdf"]


# ── the atom carries EVERY author, not just `who_id`'s first one ─────────────────

def test_the_atom_records_all_authors_with_their_registry_ids(conn, fake_embedder, monkeypatch):
    """`who_id` is the FIRST author and 67 of 82 live paper atoms do not even have that — they
    carry the `paper-authors:{paper_id}` placeholder because Semantic Scholar never resolved the
    work. So without this list the coauthors of a saved paper reach nothing, and the atom is the
    only place a reader can look: the snapshot markdown renders names for DISPLAY, and re-deriving
    them from a presentation string is not a data path.

    Registry ids stay under separate keys because `scholar:` and `openalex:` are different
    namespaces over the same person."""
    import json
    _stub_fulltext(monkeypatch, None)
    paper = {**_PAPER, "authors": [
        {"authorId": "111", "name": "Alice Researcher"},
        {"name": "Bob Coauthor", "openalexId": "A5043841592", "position": "middle",
         "orcid": "https://orcid.org/0000-0002-4027-364X"},   # bared on the way in
        {"authorId": None, "name": "Nameless Contributor"},          # id-less, still recorded
        {"authorId": "999", "name": "   "},                          # no name → not a person
    ]}
    ip.atomize_paper(conn, fake_embedder, paper)

    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()[0])
    assert payload["authors"] == [
        {"name": "Alice Researcher", "scholar_id": "111"},
        {"name": "Bob Coauthor", "openalex_id": "A5043841592",
         "orcid": "0000-0002-4027-364X", "position": "middle"},
        {"name": "Nameless Contributor"}]


def test_a_hyperauthored_paper_is_capped_not_stored_whole(conn, fake_embedder, monkeypatch):
    """HEP papers carry thousands of authors and a user saving one is not vouching for thousands
    of people. A BOUND on what the atom records, not a judgement about who matters."""
    import json
    _stub_fulltext(monkeypatch, None)
    paper = {**_PAPER, "authors": [{"authorId": str(i), "name": f"Author {i}"}
                                   for i in range(3000)]}
    ip.atomize_paper(conn, fake_embedder, paper)

    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()[0])
    assert len(payload["authors"]) == ip.MAX_ATOM_AUTHORS


# ── upgrade_to_fulltext: the abstract-only atom a reader opened ───────────────────
# The gesture these protect: a scholar Oracle's back catalogue lands abstract-only, and the FIRST
# open of one of those papers is what goes and gets the PDF. Before 2026-09-11 there was no second
# act at all — Policy B skips a present paper before the fetch, so nothing could ever deepen it.

def _abstract_only(conn, embedder, monkeypatch, **over):
    """Mint the atom the scholar footprint writes: abstract, no PDF pulled."""
    paper = {**_PAPER, **over}
    minted = ip.atomize_paper(conn, embedder, paper, fulltext=None)
    assert minted == ip.paper_atom_id(paper)
    return paper


def _payload(conn, atom_id=_ATOM):
    import json
    return json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()["payload"])


def test_upgrade_promotes_partial_to_complete(conn, fake_embedder, monkeypatch):
    _abstract_only(conn, fake_embedder, monkeypatch)
    assert _payload(conn)["body_state"] == "partial"
    assert _payload(conn)["has_fulltext"] is False

    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is True

    pay = _payload(conn)
    assert pay["body_state"] == "complete"
    assert pay["has_fulltext"] is True
    raw = conn.execute("SELECT raw_ref FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()["raw_ref"]
    from pipeline.kb.raw_store import read_snapshot
    body = read_snapshot(raw)
    assert "## Full text" in body and "INTRODUCTION." in body
    assert _PAPER["abstract"] in body            # the abstract SURVIVES the rewrite


def test_upgrade_rebuilds_the_paper_without_parsing_the_snapshot(conn, fake_embedder, monkeypatch):
    """The Paper handed to `resolve_fulltext` comes from the payload, so the upgrade can reach the
    SAME pdf mirrors the original mint would have — including a finder's `openAccessPdf`, which is
    the only route to a body for the works S2 has never heard of."""
    _abstract_only(conn, fake_embedder, monkeypatch,
                   externalIds={"DOI": "10.5281/zenodo.1"},
                   paperId="DOI:10.5281/zenodo.1",
                   openAccessPdf={"url": "https://zenodo.org/records/1/files/p.pdf"})
    seen = {}
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: seen.update(paper) or _FULLTEXT)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, "paper:DOI:10.5281/zenodo.1") is True
    assert seen["openAccessPdf"] == {"url": "https://zenodo.org/records/1/files/p.pdf"}
    assert seen["title"] == _PAPER["title"]
    assert seen["abstract"] == _PAPER["abstract"]
    # Translated BACK into S2 field names: `atom_authors` stored `scholar_id`, a Paper says
    # `authorId`. The id-less contributor survives by name, as it does at mint.
    assert seen["authors"] == [{"name": "Alice Researcher", "authorId": "111"},
                               {"name": "Bob Coauthor", "authorId": "222"},
                               {"name": "Nameless Contributor"}]


def test_upgrade_replaces_chunks_rather_than_adding_to_them(conn, fake_embedder, monkeypatch):
    """The abstract-only chunks must not linger beside the full-text ones — a stale chunk is a
    second retrievable copy of the same atom."""
    _abstract_only(conn, fake_embedder, monkeypatch)
    before = conn.execute("SELECT COUNT(*) c FROM chunks WHERE atom_id=?", (_ATOM,)).fetchone()["c"]
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)
    ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM)
    after = conn.execute("SELECT COUNT(*) c FROM chunks WHERE atom_id=?", (_ATOM,)).fetchone()["c"]
    assert after > before                                    # the full body really is chunked
    texts = [r["text"] for r in conn.execute(
        "SELECT text FROM chunks WHERE atom_id=? ORDER BY seq", (_ATOM,))]
    assert sum("INTRODUCTION." in t for t in texts) >= 1
    fts = conn.execute("SELECT COUNT(*) c FROM chunks_fts WHERE atom_id=?", (_ATOM,)).fetchone()["c"]
    assert fts == after                                      # FTS stayed in sync with the replace


def test_upgrade_preserves_identity_and_never_demotes(conn, fake_embedder, monkeypatch):
    """Identity is copied from the stored row, so no PDF can move who_id/when_ts/entry_mode."""
    _abstract_only(conn, fake_embedder, monkeypatch)
    cols = "who_id, when_ts, when_precision, source_url, description, entry_mode"
    before = dict(conn.execute(f"SELECT {cols} FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone())
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)
    ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM)
    after = dict(conn.execute(f"SELECT {cols} FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone())
    assert before == after

    # ONE-WAY: a complete atom is never re-fetched, and never walked back to partial.
    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda paper: pytest.fail("a complete paper must not re-fetch"))
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is False
    assert _payload(conn)["body_state"] == "complete"


def test_upgrade_that_finds_no_pdf_returns_the_abstract_untouched(conn, fake_embedder, monkeypatch):
    _abstract_only(conn, fake_embedder, monkeypatch)
    before = dict(conn.execute(
        "SELECT raw_ref, raw_hash, version FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone())
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is False
    after = dict(conn.execute(
        "SELECT raw_ref, raw_hash, version FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone())
    assert before == after                       # no body written, no version bump
    assert _payload(conn)["body_state"] == "partial"
    assert _payload(conn)["fulltext_tried_at"]   # ...but the attempt IS recorded


def test_a_stamped_attempt_is_not_re_paid_on_the_next_open(conn, fake_embedder, monkeypatch):
    """The stamp exists so a paper with no open PDF does not re-download on every single open."""
    _abstract_only(conn, fake_embedder, monkeypatch)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM)

    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda paper: pytest.fail("a stamped paper must not re-fetch"))
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is False


def test_a_stale_stamp_is_retried(conn, fake_embedder, monkeypatch):
    """A mirror can appear later, so the stamp is a TTL and not a tombstone."""
    import json
    from datetime import date, timedelta
    _abstract_only(conn, fake_embedder, monkeypatch)
    stale = (date.today() - timedelta(days=ip._FULLTEXT_RETRY_DAYS + 1)).isoformat()
    conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?",
                 (json.dumps({**_payload(conn), "fulltext_tried_at": stale}), _ATOM))
    conn.commit()
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is True
    assert "fulltext_tried_at" not in _payload(conn)     # cleared once it succeeded


def test_no_body_builds_no_embedder(conn, fake_embedder, monkeypatch):
    """Constructing an embedder needs an API key and a network, so the common case — an open of a
    paper whose PDF is not reachable — must never reach for one."""
    _abstract_only(conn, fake_embedder, monkeypatch)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    assert ip.upgrade_to_fulltext(
        conn, lambda: pytest.fail("no embedder may be built when no body resolved"), _ATOM) is False


def test_a_failed_write_returns_the_abstract_and_is_retried_now(conn, fake_embedder, monkeypatch):
    """Fail-safe: this runs inside open(), so an embed outage returns the body we already hold.
    It is NOT stamped — the PDF was reachable and only the write failed, so the next open should
    retry immediately rather than wait out the TTL."""
    _abstract_only(conn, fake_embedder, monkeypatch)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)

    def _boom(*a, **k):
        raise RuntimeError("embed endpoint down")
    monkeypatch.setattr(ip, "store_atom", _boom)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is False
    assert _payload(conn)["body_state"] == "partial"
    assert "fulltext_tried_at" not in _payload(conn)

    monkeypatch.undo()
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _FULLTEXT)
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is True


def test_upgrade_ignores_a_non_paper_atom(conn, fake_embedder, monkeypatch):
    _abstract_only(conn, fake_embedder, monkeypatch)
    conn.execute("UPDATE atoms SET source_type='blog' WHERE atom_id=?", (_ATOM,))
    conn.commit()
    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda paper: pytest.fail("only papers deepen on open"))
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, _ATOM) is False
    assert ip.upgrade_to_fulltext(conn, lambda: fake_embedder, "paper:nope") is False
