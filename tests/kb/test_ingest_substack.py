"""ingest_substack — one Substack post somebody referenced → an opinion atom
(`substack_atom_from_url`). The archive walk's per-post builder is covered with the walk
(`test_archive_adapters.py`).

Offline: the per-post-body fetch is monkeypatched, so these prove the wiring — the atom build,
policy-B dedup, the paywalled skip, reader/app/custom-domain urls — not the live scrape.

Real-API shape verified live: the archive list carries NO body_html (metadata +
`truncated_body_text` only); the full body comes from `_fetch_full_post(base, slug, {})`, which
works cookie-less for PUBLIC posts. The mocks mirror that (archive posts carry no body).
"""
from __future__ import annotations


import pytest

from pipeline.kb import ingest_substack as fp
from pipeline.kb import schema
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import publication as substack_publication


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


_PUB = "https://carol.substack.com"
_BODY = ("<h2>Intro</h2><p>Autonomous agents compose small tools into larger systems.</p>"
         "<p>See <a href='https://arxiv.org/abs/1234'>the paper</a>.</p>")


def _post(pid, *, title="Scaling Laws", audience="everyone", slug="scaling-laws"):
    """One ARCHIVE-list metadata dict — no body_html (matches the real endpoint)."""
    return {"id": pid, "title": title, "subtitle": "a deep dive",
            "post_date": "2026-05-01T00:00:00Z", "audience": audience, "slug": slug,
            "canonical_url": f"{_PUB}/p/{slug}", "wordcount": 900, "body_html": None}


BLOCKED = object()   # sentinel: this slug's fetch is STOPPED (Cloudflare / transport), not empty


def _patch(monkeypatch, posts, *, bodies=None):
    """Patch the archive list + the per-post full-body fetch. `bodies` maps slug→body_html
    (None → the per-post fetch returns None, i.e. no full body; `BLOCKED` → the fetch raises
    `SubstackFetchError`, i.e. we never learned whether the post has a body); default = _BODY
    for every post."""
    monkeypatch.setattr(substack_publication, "fetch_all_posts", lambda base, since=None: [dict(p) for p in posts])

    def _full(base, slug, cookies):
        b = _BODY if bodies is None else bodies.get(slug)
        if b is BLOCKED:
            raise substack_contracts.SubstackFetchError(f"cloudflare challenge for {slug!r}")
        return {"body_html": b} if b else None
    monkeypatch.setattr(substack_publication, "fetch_full_post", _full)


# ── a STOPPED archive listing ingests nothing (fail-safe on an incomplete inventory) ──


# ── the happy path: one public post → one full-body footprint atom ────────────────


# ── D: VLM describes inline images → the description becomes searchable body text ──


# ── policy B: the second run skips already-ingested posts BEFORE any re-fetch ──────


# ── paywalled + no-body are SKIPPED and counted (never a partial atom) ─────────────


# ── a BLOCKED fetch is counted apart from a body-less post, and stays retryable ────
# The whole point of the three-verdict split: "Cloudflare stopped us" and "this post is a
# podcast" must not be the same `None`, or a throttled run reports itself as a thin archive.


# ── the `limit` bound caps NEW atoms ──────────────────────────────────────────────


# ── the shared key: a saved post + the SAME footprint post = ONE atom ─────────────


# ── attribution: resolve unifies the footprint author into the Oracle's canonical ─


# ── substack_atom_from_url: the Stage-5 link-dispatch twin (ONE referenced public post) ──
# Same repo→atom shape as the whole-publication footprint above, different ENTRY: the Oracle
# *referenced* this post (entry_mode 'author_referenced'), it isn't their own archive.

def _full_post(pid, *, slug="essay", audience="everyone", title="Referenced Essay",
               body=None, host="https://joe.substack.com"):
    """The PER-POST endpoint shape (`_fetch_full_post`) — carries the numeric id + the full body."""
    return {"id": pid, "title": title, "slug": slug, "audience": audience,
            "post_date": "2026-04-02T00:00:00Z", "wordcount": 1200,
            "canonical_url": f"{host}/p/{slug}", "body_html": body or _BODY}


def _patch_post(monkeypatch, full):
    monkeypatch.setattr(substack_publication, "fetch_full_post", lambda base, slug, cookies: full)


def test_atom_from_url_mints_referenced_post(conn, monkeypatch):
    _patch_post(monkeypatch, _full_post(700))
    atom_id = fp.substack_atom_from_url(conn, "https://joe.substack.com/p/essay")
    assert atom_id == "substack:700"
    row = conn.execute("SELECT who_id, what_kind, entry_mode, source_type FROM atoms "
                       "WHERE atom_id='substack:700'").fetchone()
    assert row["who_id"] == "substack:joe" and row["what_kind"] == "opinion"
    assert row["entry_mode"] == "author_referenced" and row["source_type"] == "substack"


_IMG_BODY = _BODY + "<p><img src='https://substack-post-media.s3.amazonaws.com/c.png'></p>"


def test_a_referenced_post_reads_no_image(conn, monkeypatch, ocr):
    """An Oracle linked it, so it was pulled in automatically: its images are never read."""
    _patch_post(monkeypatch, _full_post(710, body=_IMG_BODY))
    assert fp.substack_atom_from_url(conn, "https://joe.substack.com/p/essay")
    assert ocr.calls == []


def test_a_hopper_saved_post_reads_its_images(conn, monkeypatch, ocr):
    _patch_post(monkeypatch, _full_post(711, body=_IMG_BODY))
    assert fp.substack_atom_from_url(conn, "https://joe.substack.com/p/essay",
                                     entry_mode="user-saved")
    assert len(ocr.calls) == 1


def test_atom_from_url_not_a_post(conn):
    # A profile / about page (no /p/slug) is not a post → None (nothing to mint or vouch).
    assert fp.substack_atom_from_url(conn, "https://joe.substack.com/about") is None


def test_atom_from_url_paywalled_skipped(conn, monkeypatch):
    _patch_post(monkeypatch, _full_post(701, audience="only_paid"))   # preview-only → never stored
    assert fp.substack_atom_from_url(conn, "https://joe.substack.com/p/paid") is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_atom_from_url_policy_b_never_clobbers(conn, monkeypatch):
    # A post already present (e.g. from a curation SAVE) must NOT be re-rendered/overwritten — the
    # referenced path returns its id for the vouch and leaves the existing provenance intact.
    schema.upsert_atom(conn, {"atom_id": "substack:702", "source_type": "substack",
                              "entry_mode": "user-saved", "raw_hash": "seed"})
    _patch_post(monkeypatch, _full_post(702, slug="dup"))
    atom_id = fp.substack_atom_from_url(conn, "https://joe.substack.com/p/dup")
    assert atom_id == "substack:702"
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='substack:702'"
                        ).fetchone()["entry_mode"] == "user-saved"          # untouched


def test_resolve_reader_url(monkeypatch):
    # The platform reader-url carries only a global post id; it 302-redirects to the canonical
    # {pub}/p/{slug} (verified live), which is what the resolver follows.
    import requests

    class _R:
        url = "https://a16zcrypto.substack.com/p/agents-real"
    monkeypatch.setattr(requests, "head", lambda url, **k: _R())
    assert fp._resolve_reader_url("https://substack.com/home/post/p-194563373") == \
        "https://a16zcrypto.substack.com/p/agents-real"

    class _R2:                                   # a redirect that does NOT land on a /p/ post → None
        url = "https://substack.com/home"
    monkeypatch.setattr(requests, "head", lambda url, **k: _R2())
    assert fp._resolve_reader_url("https://substack.com/home/post/p-1") is None


def test_atom_from_url_reader_url_resolved_then_minted(conn, monkeypatch):
    # A reader-url resolves to an a16zcrypto post → minted under THAT publication's who_id, proving
    # the canonical pub (not the reader host `substack.com`) is what attribution keys on.
    monkeypatch.setattr(fp, "_resolve_reader_url",
                        lambda u: "https://a16zcrypto.substack.com/p/agents-real")
    _patch_post(monkeypatch, _full_post(800, slug="agents-real",
                                        host="https://a16zcrypto.substack.com"))
    atom_id = fp.substack_atom_from_url(conn, "https://substack.com/home/post/p-194563373")
    assert atom_id == "substack:800"
    assert conn.execute("SELECT who_id FROM atoms WHERE atom_id='substack:800'"
                        ).fetchone()["who_id"] == "substack:a16zcrypto"


# ── Which HOSTS the direct post parser accepts, and why that is not the host boundary ────────

def test_atom_from_url_mints_a_custom_domain_substack(conn, monkeypatch):
    """`_SUBSTACK_POST_RE` accepts ANY host with `/p/{slug}`, deliberately. A Substack on its own
    domain is indistinguishable from a generic blog without a fetch, so `classify_link` cannot
    sniff one and it arrives here only through Hopper's explicit `kind_hint`. `derive
    .substack_entity_id` keys it on the custom host — a host test HERE would delete that path."""
    _patch_post(monkeypatch, _full_post(900, slug="rate-cuts", host="https://noahpinion.blog"))
    atom_id = fp.substack_atom_from_url(conn, "https://noahpinion.blog/p/rate-cuts")
    assert atom_id == "substack:900"
    assert conn.execute("SELECT who_id FROM atoms WHERE atom_id='substack:900'"
                        ).fetchone()["who_id"] == "substack:noahpinion.blog"


def test_atom_from_url_on_a_host_that_serves_no_post_api_writes_nothing(conn, monkeypatch):
    """The lookalike-host fail-safe. `classify_link` does not route `not-substack.com` here
    (tests/kb/test_link_router.py), and if a hint forces it, the per-post fetch raises and the
    whole call SKIPS — no atom, no entity, nothing marked seen."""

    def _blocked(base, slug, cookies):
        raise substack_contracts.SubstackFetchError(f"no post api at {base}")
    monkeypatch.setattr(substack_publication, "fetch_full_post", _blocked)

    assert fp.substack_atom_from_url(conn, "https://not-substack.com/p/not-a-post") is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0


# ── the app's share link ──

def test_an_app_share_link_resolves_to_its_post_without_a_network_hop(conn, monkeypatch):
    """`open.substack.com/pub/{pub}/p/{slug}` — what the iOS share sheet copies — names the
    publication and slug outright. It never matched the post pattern, so it was never saved."""
    seen: list = []
    monkeypatch.setattr(substack_publication, "fetch_full_post",
                        lambda base, slug, cookies: seen.append((base, slug)) or None)
    monkeypatch.setattr(fp, "_resolve_reader_url",
                        lambda u: (_ for _ in ()).throw(AssertionError("no redirect hop needed")))

    fp.substack_atom_from_url(conn, "https://open.substack.com/pub/signull/p/the-end-of-goldfish-ai"
                               "?r=qq68&utm_medium=ios")

    assert seen == [("https://signull.substack.com", "the-end-of-goldfish-ai")]
