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


def test_atomize_who_id_override(conn, fake_embedder, monkeypatch):
    # A driver may pass an explicit who_id; it wins over the derived first-author id.
    _stub_fulltext(monkeypatch, _FULLTEXT)
    ip.atomize_paper(conn, fake_embedder, _PAPER, who_id="scholar:override")
    row = conn.execute("SELECT who_id FROM atoms WHERE atom_id=?", (_ATOM,)).fetchone()
    assert row["who_id"] == "scholar:override"


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
    """Patch the `requests.get` that `_fetch_s2_paper` imports at call time."""
    import requests

    def _get(url, **kw):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc
    monkeypatch.setattr(requests, "get", _get)


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


def test_the_fallback_fires_only_when_the_title_is_missing(monkeypatch):
    """The cost gate. A caller that supplied `known=` (the frontier path) or an S2 that answered
    both leave a title behind, and neither pays for a request."""
    called = []
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    monkeypatch.setattr(ip, "_zenodo_metadata", lambda paper: called.append(paper) or None)
    ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True,
                      known={"title": "The finder already knew this"})
    assert called == []


def test_an_s2_404_on_a_zenodo_doi_no_longer_mints_untitled(monkeypatch):
    """The regression this closes, end to end through `paper_from_url`. Two of three paper callers
    pass no `known=`, and `ingest_x_footprint` mints `author_referenced` atoms — human-attested —
    so the bad row landed in the tier the KB is built on."""
    monkeypatch.setattr(ip, "_fetch_s2_paper", lambda lookup: (None, ip.FETCH_ABSENT))
    _zenodo(monkeypatch, _ZenodoResp(200, _ZENODO_REC))
    paper = ip.paper_from_url("https://doi.org/10.5281/zenodo.21921441", enrich=True)
    assert paper["title"].startswith("Continuous Memory")
    assert paper["abstract"] and paper["authors"]
    assert "Untitled" not in ip.paper_to_markdown_full(paper, None)


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
