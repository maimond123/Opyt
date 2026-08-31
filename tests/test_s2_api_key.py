"""S2_API_KEY — the optional Semantic Scholar credential, and where it may and may not go.

WHY THIS EXISTS. Six independent call sites hit Semantic Scholar and none of them shared a
client, so a key wired into one would work depending on which code path you entered. These pin
the two halves that matter: the key reaches EVERY S2 request, and it reaches NOTHING ELSE.

The second half is the security one. `resolve_fulltext` downloads PDFs from arXiv, from
open-access mirrors, and from whatever host a blog links to. Those requests must never carry an
S2 credential — a shared header dict is precisely how that leak gets introduced.
"""
from __future__ import annotations

import pytest

from opyt_core import keys as core_keys
from pipeline import credentials
from pipeline.kb import ingest_papers as ip


@pytest.fixture()
def no_key(monkeypatch):
    monkeypatch.setattr(credentials, "get_credential", lambda s: None)


@pytest.fixture()
def with_key(monkeypatch):
    monkeypatch.setattr(credentials, "get_credential",
                        lambda s: "s2-test-key" if s == "semanticscholar" else None)


# ── the helper ───────────────────────────────────────────────────────────────────
def test_absent_key_still_returns_usable_headers(no_key):
    """Fail-safe: no key must degrade to plain headers, never raise and never block the call.
    Everything still works without a key — it is throttled, not broken."""
    h = credentials.s2_headers()
    assert h == {"User-Agent": credentials.S2_USER_AGENT}
    assert "x-api-key" not in h


def test_key_is_attached_as_x_api_key(with_key):
    """S2 authenticates with `x-api-key`, NOT `Authorization: Bearer`. A Bearer header is
    silently IGNORED by S2 — the request succeeds anonymously and you keep getting 429s with a
    valid key in hand, which is close to impossible to debug from the outside."""
    h = credentials.s2_headers()
    assert h["x-api-key"] == "s2-test-key"
    assert "Authorization" not in h


def test_headers_are_a_plain_dict_both_http_styles_accept(with_key):
    """Four call sites use `requests`, two use `urllib`. Both take a plain dict, so the helper
    must not return a requests-specific object."""
    assert type(credentials.s2_headers()) is dict


# ── it reaches the S2 request ────────────────────────────────────────────────────
def test_the_atom_kb_s2_fetch_sends_the_key(monkeypatch, with_key):
    seen = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"title": "T"}

    def _get(url, **kw):
        seen["url"], seen["headers"] = url, kw.get("headers") or {}
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "get", _get)

    ip._fetch_s2_paper("arXiv:1706.03762")
    assert "semanticscholar.org" in seen["url"]
    assert seen["headers"].get("x-api-key") == "s2-test-key"


# ── it must NOT reach anything else ──────────────────────────────────────────────
def test_pdf_download_never_carries_the_s2_key(monkeypatch, with_key):
    """THE credential-leak guard. A PDF is fetched from arXiv, an OA mirror, or an arbitrary
    blog host — none of which issued this key. Sending it there hands a third-party site a
    working credential."""
    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=0):
            return iter([b"%PDF-1.4 fake"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _get(url, **kw):
        seen["url"], seen["headers"] = url, kw.get("headers") or {}
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "get", _get)

    ip._download_pdf("https://some-random-blog.example.com/paper.pdf")
    assert "x-api-key" not in seen["headers"], (
        f"S2 credential leaked to {seen['url']} — PDF downloads must use _PDF_UA")


def test_pdf_headers_and_s2_headers_are_not_the_same_object(with_key):
    """Structural half of the guard above: if one dict served both, any future key added to it
    would leak by construction and no behavioural test would necessarily catch it."""
    assert "x-api-key" not in ip._PDF_UA
    assert ip._PDF_UA is not credentials.s2_headers()


# ── registered so onboarding can offer it ────────────────────────────────────────
def test_s2_key_is_registered_everywhere_onboarding_looks():
    """A key the user can set but no surface mentions is a key nobody sets. All three registries
    have to know about it: the service map, `opyt-keys --list`, and the .env template."""
    assert credentials.SERVICES["semanticscholar"] == "S2_API_KEY"
    assert "S2_API_KEY" in core_keys.KNOWN


# The preflight test (both optional keys warn with their signup URL) was deleted 2026-08-29 with
# its subject: `lifecycle.check_prerequisites` had no production caller (`retired-lifecycle-
# preflight` guard). Signup-URL rendering is pinned registry-side by
# `credential-metadata-lives-in-the-registry`.


# ── validation ───────────────────────────────────────────────────────────────────
def test_a_429_during_validation_is_not_reported_as_a_bad_key(monkeypatch):
    """An anonymous burst produces 429 too, so a 429 cannot distinguish a bad key from a busy
    moment. Calling it invalid sends the user to regenerate a perfectly good key."""
    class _Resp:
        status_code = 429

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp())

    ok, msg = credentials.validate_credential("semanticscholar", "s2-test-key")
    assert ok is True
    assert "does NOT mean it is invalid" in msg


@pytest.mark.parametrize("status,expect_ok", [(200, True), (401, False), (403, False), (500, False)])
def test_validation_maps_status_to_verdict(monkeypatch, status, expect_ok):
    class _Resp:
        status_code = None

        def json(self):
            return {"title": "Attention is All you Need"}

    _Resp.status_code = status
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp())

    ok, _ = credentials.validate_credential("semanticscholar", "s2-test-key")
    assert ok is expect_ok
