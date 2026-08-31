"""link_router.classify_link_deep — the bounded structural fallback for a paper on a host
`_PAPER_HOSTS` doesn't list. Pure unit tests: `requests.get` is monkeypatched, so these prove the
sniffing logic (Content-Type, citation_doi rewrite, citation_* confirm, size cap, fail-safe on
error) without any real network call.
"""
from __future__ import annotations

import requests

from pipeline.kb import link_router as lr


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
    html = b'<html><head><meta name="citation_title" content="Some Paper"></head></html>'
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    assert lr.classify_link_deep("https://ssrn.com/abstract=123") == (
        "paper", "https://ssrn.com/abstract=123", None)


def test_plain_blog_page_is_not_a_paper(monkeypatch):
    html = b"<html><head><title>My Blog Post</title></head><body>hello</body></html>"
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[html]))
    assert lr.classify_link_deep("https://example.com/post") is None


def test_non_html_non_pdf_response_is_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "image/png"}))
    assert lr.classify_link_deep("https://example.com/photo.png") is None


def test_http_error_status_is_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, _Resp(status=404))
    assert lr.classify_link_deep("https://example.com/gone") is None


def test_network_failure_is_fail_safe_not_a_paper(monkeypatch):
    _stub_get(monkeypatch, requests.ConnectionError("refused"))
    assert lr.classify_link_deep("https://example.com/down") is None


def test_oversized_page_without_a_citation_tag_gives_up(monkeypatch):
    huge = b"<html>" + b"x" * (lr._DEEP_PROBE_MAX_BYTES + 1) + b"</html>"
    _stub_get(monkeypatch, _Resp(headers={"Content-Type": "text/html"}, chunks=[huge]))
    assert lr.classify_link_deep("https://example.com/huge") is None
