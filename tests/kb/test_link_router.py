"""link_router.classify_link_deep — the bounded structural fallback for a paper on a host
`_PAPER_HOSTS` doesn't list. Pure unit tests: `requests.get` is monkeypatched, so these prove the
sniffing logic (Content-Type, citation_doi rewrite, citation_* confirm, size cap, fail-safe on
error) without any real network call.
"""
from __future__ import annotations

import pytest
import requests

from pipeline.kb import link_router as lr, schema


class _Resp:
    """Minimal stand-in for `requests.Response` — just what `classify_link_deep` reads."""

    def __init__(self, status=200, headers=None, chunks=(b"",)):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)

    def iter_content(self, chunk_size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _no_openalex(monkeypatch):
    """Silence the SECOND source. `classify_link_deep` now falls through to OpenAlex whenever the
    page itself said nothing, so a test about the PAGE has to state what the index said too."""
    from pipeline.kb import ingest_papers
    monkeypatch.setattr(ingest_papers, "_openalex_doi_by_url", lambda url: None)


def _openalex_says(monkeypatch, doi, seen=None):
    from pipeline.kb import ingest_papers
    def _f(url):
        if seen is not None:
            seen.append(url)
        return doi
    monkeypatch.setattr(ingest_papers, "_openalex_doi_by_url", _f)


def _stub_get(monkeypatch, resp_or_exc):
    def _get(url, **kw):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc
    monkeypatch.setattr(requests, "get", _get)


def test_pdf_content_type_is_a_paper(monkeypatch):
    # The content_type is carried through so a mint can key on the FACT even when the url's shape
    # gives no hint (a `/download?id=1` redirect — see test_ingest_papers.py for the mint side).
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "application/pdf"}))
    assert lr.classify_link_deep("https://example.com/download?id=1") == (
        "paper", "https://example.com/download?id=1", "application/pdf")


def test_citation_doi_meta_rewrites_to_doi_org(monkeypatch):
    html = b'<html><head><meta name="citation_doi" content="10.1038/s41586-021-03819-2"></head></html>'
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    assert lr.classify_link_deep("https://nature.com/articles/s41586-021-03819-2") == (
        "paper", "https://doi.org/10.1038/s41586-021-03819-2", None)


def test_other_citation_meta_confirms_paper_without_a_mint_url(monkeypatch):
    """Scholarly markup, no DOI, and an index that does not know the page either — the ONE state
    in which a `paper` with no mintable id is still the right answer, because `article` would file
    a paper as a blog post. `_no_openalex` is load-bearing since 2026-09-16: the index is asked
    first now, so a test about the PAGE has to say what the index said."""
    html = b'<html><head><meta name="citation_title" content="Some Paper"></head></html>'
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    _no_openalex(monkeypatch)
    assert lr.classify_link_deep("https://example.edu/handle/123") == (
        "paper", "https://example.edu/handle/123", None)


def test_scholarly_markup_asks_the_index_before_settling_for_a_nameless_paper(monkeypatch):
    """The ordering fix. A page carrying `citation_*` but no DOI used to return `paper` outright,
    which skipped the only source that could NAME it — 8 of the 124 refused urls in the Set A+B
    sweeps, `repositorio.unal.edu.co/handle/unal/81443` (ResNet) among them."""
    html = b'<html><head><meta name="citation_author" content="K He"></head></html>'
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    _openalex_says(monkeypatch, "10.1109/cvpr.2016.90")
    assert lr.classify_link_deep("https://repositorio.unal.edu.co/handle/unal/81443") == (
        "paper", "https://doi.org/10.1109/cvpr.2016.90", None)


def test_plain_blog_page_is_not_a_paper(monkeypatch):
    html = b"<html><head><title>My Blog Post</title></head><body>hello</body></html>"
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    _no_openalex(monkeypatch)
    assert lr.classify_link_deep("https://example.com/post") is None


def test_non_html_non_pdf_response_is_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "image/png"}))
    assert lr.classify_link_deep("https://example.com/photo.png") is None


def test_http_error_status_is_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, _Resp(status=404))
    _no_openalex(monkeypatch)
    assert lr.classify_link_deep("https://example.com/gone") is None


def test_network_failure_is_fail_safe_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, requests.ConnectionError("refused"))
    _no_openalex(monkeypatch)
    assert lr.classify_link_deep("https://example.com/down") is None


def test_oversized_page_without_a_citation_tag_gives_up(monkeypatch):
    huge = b"<html>" + b"x" * (lr._DEEP_PROBE_MAX_BYTES + 1) + b"</html>"
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[huge]))
    _no_openalex(monkeypatch)
    assert lr.classify_link_deep("https://example.com/huge") is None


# ── the second source: what OpenAlex records at a url the page will not show us ──────────────
#
# Measured 2026-09-11 over 250 pasted urls: 96 were refused by everything above, and 22 of the 37
# the probe could not crack answered 4xx to BOTH our User-Agent and a browser's. Those pages were
# never going to be read. Asking the index recovers 59 of the 96.

def test_a_publisher_that_refuses_the_fetch_is_still_resolved(monkeypatch):
    """The big one. Elsevier, MDPI, RSC, Cell, SSRN, PMC and ResearchGate all answer 403 to a
    server fetch, and a browser User-Agent changes nothing — so a 4xx says nothing about whether
    this is a paper, and must not be read as 'no'."""
    seen = []
    _stub_get(monkeypatch, _Resp(status=403))
    _openalex_says(monkeypatch, "10.1109/cvpr.2016.90", seen)
    assert lr.classify_link_deep("https://www.sciencedirect.com/science/article/pii/S1359645413005430") == (
        "paper", "https://doi.org/10.1109/cvpr.2016.90", None)
    assert seen == ["https://www.sciencedirect.com/science/article/pii/S1359645413005430"]


def test_a_repository_page_that_declares_nothing_is_still_resolved(monkeypatch):
    """`repositorio.unal.edu.co/handle/unal/81443` fetches fine, declares no identifier, and is
    ResNet. The page is not the only thing that knows what it is holding."""
    html = b"<html><head><title>Repositorio</title></head><body>a record</body></html>"
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    _openalex_says(monkeypatch, "10.1109/cvpr.2016.90")
    assert lr.classify_link_deep("https://repositorio.unal.edu.co/handle/unal/81443") == (
        "paper", "https://doi.org/10.1109/cvpr.2016.90", None)


def test_the_page_still_outranks_the_index(monkeypatch):
    """Order of authority. A `citation_doi` is the publisher naming its OWN paper; the index is a
    third party's record of where a copy lives. The tag wins whenever both exist."""
    html = b'<html><head><meta name="citation_doi" content="10.1038/real"></head></html>'
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    _openalex_says(monkeypatch, "10.9999/wrong")
    assert lr.classify_link_deep("https://example.com/article")[1] == "https://doi.org/10.1038/real"


def test_a_successful_non_article_response_costs_no_index_call(monkeypatch):
    """A 200 that hands back a PNG is a real answer — there is no article here — unlike a 4xx,
    which is a refusal. Believing the former is what keeps this from asking about every image."""
    from pipeline.kb import ingest_papers
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "image/png"}))
    monkeypatch.setattr(ingest_papers, "_openalex_doi_by_url",
                        lambda url: pytest.fail("a 200 non-article must not spend a lookup"))
    assert lr.classify_link_deep("https://example.com/photo.png") is None


# ── classify_link: the FREE host sniff, and where a provider's name stops being one ──────────
# The sniff is what the docstring calls a fact, so it has to be a hostname test and not a
# substring one. `"substack.com" in netloc` also fires on `not-substack.com` and on
# `github.com.evil.example`, which routed an arbitrary host into a provider adapter — and, through
# `ingest_x_footprint`'s shared caller, into the "this link is dispatchable" substance filter.

def test_provider_hosts_and_their_subdomains_still_sniff():
    assert lr.classify_link("https://github.com/karpathy/nanogpt") == "github"
    assert lr.classify_link("https://www.github.com/karpathy/nanogpt") == "github"
    assert lr.classify_link("https://carol.substack.com/p/essay") == "substack"
    assert lr.classify_link("https://substack.com/home/post/p-1") == "substack"
    assert lr.classify_link("https://arxiv.org/abs/2401.00001") == "paper"
    assert lr.classify_link("https://www.biorxiv.org/content/10.1101/1") == "paper"
    assert lr.classify_link("https://pubmed.ncbi.nlm.nih.gov/12345/") == "paper"


def test_a_lookalike_host_is_not_the_provider():
    # Left-hand extension: a different registrable domain that merely CONTAINS the provider's.
    assert lr.classify_link("https://not-substack.com/p/not-a-post") is None
    assert lr.classify_link("https://notgithub.com/o/r") is None
    assert lr.classify_link("https://myarxiv.org/abs/1") is None
    # Right-hand extension: the provider's name as a LABEL under somebody else's domain.
    assert lr.classify_link("https://github.com.evil.example/o/r") is None
    assert lr.classify_link("https://arxiv.org.evil.example/abs/1") is None


def test_a_port_or_uppercase_host_does_not_defeat_the_sniff():
    """`.hostname` normalizes case and strips the port; the old `.netloc` test carried both."""
    assert lr.classify_link("https://GitHub.COM/o/r") == "github"
    assert lr.classify_link("http://arxiv.org:8080/abs/1") == "paper"


@pytest.mark.parametrize("url", [
    "https://ssrn.com/abstract=3482150",
    "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3482150",
    "https://zenodo.org/records/21921441",
    "https://www.alphaxiv.org/abs/2005.14165",
    "https://huggingface.co/papers/2005.14165",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8371605/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/",
])
def test_the_hosts_that_name_a_paper_without_a_doi_in_the_url(url):
    """Added 2026-09-11. Each of these was routed to the ARTICLE catch-all, and each answers 403
    to a server fetch (browser User-Agent included, measured), so `classify_link_deep` could not
    rescue them either — they were simply lost. Same justification as arXiv and PubMed: the url
    names the paper, it just does not spell the name as a DOI."""
    assert lr.classify_link(url) == "paper"


@pytest.mark.parametrize("url", [
    "https://huggingface.co/meta-llama/Llama-3-8B",
    "https://huggingface.co/datasets/squad",
    "https://www.ncbi.nlm.nih.gov/gene/672",
    "https://www.ncbi.nlm.nih.gov/nuccore/NM_007294",
])
def test_a_partly_papers_host_only_routes_its_papers_path(url):
    """Hugging Face is mostly models; ncbi.nlm.nih.gov is mostly sequence databases. A bare host
    entry would route all of it to the paper adapter — the same class of mis-route `_is_host`
    prevents at the host level, one level further down the url."""
    assert lr.classify_link(url) is None


def test_a_pdf_anywhere_is_still_a_paper():
    """The one non-host rule, unchanged: the extension is the fact, whatever the host."""
    assert lr.classify_link("https://example.com/papers/draft.pdf") == "paper"
    assert lr.classify_link("https://example.com/papers/draft.pdf?dl=1") == "paper"


def test_a_non_url_reference_sniffs_to_nothing():
    assert lr.classify_link("not a url at all") is None
    assert lr.classify_link("") is None


def test_a_lookalike_host_falls_through_to_the_article_catch_all():
    """The routing consequence: Hopper stops handing it to the Substack adapter and treats it as
    what it is — a page on some site."""
    assert lr.classify_reference("https://not-substack.com/p/not-a-post") == ("article", "fallback")
    # An explicit hint still reaches the Substack adapter — that is the ONLY route a custom-domain
    # Substack (noahpinion.blog/p/…) has, since no host test can identify one.
    assert lr.classify_reference("https://not-substack.com/p/x", hint="substack") == (
        "substack", "hint")


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


# ── mint_artifact: the urls whose id is not in the string ───────────────────────
#
# THE GAP THIS CLOSES. Every test above asks what a url CLASSIFIES as; none asked whether the
# thing it classifies as can actually be stored. Between 2026-08-13 and 2026-09-16 the answer for
# PubMed was no — `mint_artifact` bailed on any paper without a url-derivable id, so a headline
# advertised host returned `failed` for a month, then four more url classes were added to the same
# dead path, and 3612 green tests said nothing. A classification test cannot catch that; only a
# test that follows the url to the adapter can.

class _RecordingPapers:
    """Stand-in for `ingest_papers` inside `mint_artifact` — records what reached the adapter."""

    def __init__(self, atom_id="paper:DOI:10.1056/nejmoa2002032"):
        self.atom_id, self.from_url_calls, self.atomize_calls = atom_id, [], []

    def paper_from_url(self, url, *, enrich=True, **kw):
        # `enrich=False` is `predicted_atom_id` asking the offline question, and None is the real
        # answer for every url here. Stubbing it as anything else would hand the test the very
        # id whose absence is the thing being tested.
        if not enrich:
            return None
        self.from_url_calls.append(url)
        # `paperId` is the DETERMINISTIC one `paper_from_url` stamps after re-parsing the looked-up
        # DOI — the real `paper_atom_id` reads it first, so a placeholder here would prove nothing.
        return {"paperId": "DOI:10.1056/nejmoa2002032",
                "externalIds": {"DOI": "10.1056/nejmoa2002032"}, "title": "T"}

    def atomize_paper(self, conn, embedder, paper, **kw):
        self.atomize_calls.append(paper)
        return self.atom_id


@pytest.fixture()
def _papers(monkeypatch):
    """Swap the adapter for a recorder. Patched on the MODULE `mint_artifact` imports from, since
    it imports inside the function body."""
    rec = _RecordingPapers()
    from pipeline.kb import ingest_papers
    for name in ("paper_from_url", "atomize_paper"):
        monkeypatch.setattr(ingest_papers, name, getattr(rec, name))
    return rec


@pytest.mark.parametrize("url", [
    "https://pubmed.ncbi.nlm.nih.gov/32109013/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7096066/",
    "https://europepmc.org/article/MED/32109013",
    "https://europepmc.org/article/PMC/PMC7096066",
    "https://europepmc.org/article/PPR/PPR217527",
    "https://zenodo.org/records/3509134",
    "https://openalex.org/W2741809807",
])
def test_a_paper_whose_id_needs_a_lookup_still_reaches_the_adapter(conn, _papers, url):
    """The regression test for the month-long PubMed outage.

    `predicted_atom_id` returns None for all of these BY DESIGN — the id is knowable, just not
    from the string, and the free already-present pre-check may not pay a round trip to find out.
    That null must not be read as "not a paper"."""
    assert lr.predicted_atom_id(url, "paper") is None, "premise: the id is not in the string"
    res = lr.mint_artifact(conn, None, url, "paper", entry_mode="user-saved")
    assert _papers.from_url_calls == [url], "the adapter was never reached"
    assert res["status"] == "minted"
    # The id the ADAPTER resolved, not the null the url predicted. Returning None here would make
    # a successful save look like a failed one to every caller downstream.
    assert res["atom_id"] == "paper:DOI:10.1056/nejmoa2002032"


def test_a_second_save_of_that_paper_reports_present_and_promotes(conn, _papers):
    """The other half `aid` being None broke, and the quieter one: with no id there is nothing to
    check presence against, so a re-save re-ran the adapter and answered `minted` for an atom that
    was already there — losing the user-saved attestation `promote_atom` exists to record."""
    url = "https://pubmed.ncbi.nlm.nih.gov/32109013/"
    schema.upsert_atom(conn, {"atom_id": "paper:DOI:10.1056/nejmoa2002032",
                              "source_type": "paper", "entry_mode": "frontier",
                              "raw_hash": "h0"})
    res = lr.mint_artifact(conn, None, url, "paper", entry_mode="user-saved")
    assert res["status"] == "present"
    assert res["atom_id"] == "paper:DOI:10.1056/nejmoa2002032"
    assert _papers.atomize_calls == [], "an already-stored paper must not be re-minted"
    row = conn.execute("SELECT entry_mode FROM atoms WHERE atom_id=?",
                       ("paper:DOI:10.1056/nejmoa2002032",)).fetchone()
    assert row[0] == "user-saved", "the deposit attestation was dropped"


def test_a_junk_paper_url_spends_no_request(conn, monkeypatch):
    """The cost argument the removed bail rested on, kept honest.

    Dropping the bail means an unparseable `paper` url now reaches `paper_from_url` — which is
    fine only because that function recognises no lookup form here, so it returns None before any
    network call. If a resolver ever fires on an arbitrary url, this fails."""
    from pipeline.kb import ingest_papers
    calls = []
    monkeypatch.setattr(ingest_papers, "_looked_up_doi",
                        lambda u: calls.append(u) or None)
    url = "https://example.com/some/page"          # NOT a .pdf — that parses, and never bailed
    assert lr.predicted_atom_id(url, "paper") is None
    res = lr.mint_artifact(conn, None, url, "paper")
    assert res == {"status": "failed", "atom_id": None, "used_prefetch": False}
    assert calls == [url], "one pure-string check, no request"


def test_github_still_bails_without_a_derivable_id(conn, monkeypatch):
    """The bail was right for github and stays: there is no lookup that turns a non-repo url into
    a repo, so reaching the adapter could only waste a round trip."""
    from pipeline.kb import ingest_github
    called = []
    monkeypatch.setattr(ingest_github, "github_atom_from_url",
                        lambda *a, **k: called.append(a) or None)
    assert lr.mint_artifact(conn, None, "https://github.com/", "github")["status"] == "failed"
    assert called == []
