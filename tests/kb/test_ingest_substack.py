"""ingest_substack (Stage-5 footprint) — a confirmed Oracle's OWN archive → opinion atoms.

Offline: the archive-list + per-post-body fetches are monkeypatched, so these prove the WIRING —
full-body atom build, policy-B dedup (skip before the paid per-post fetch), paywalled/no-body
skip-and-count, the `limit` bound, the shared `substack:{post_id}` key (collision with a curation
save), entry_mode, and attribution via the upserted publication link — not the live scrape.

Real-API shape verified live 2026-07-18: the archive list carries NO body_html (metadata +
`truncated_body_text` only); the full body comes from `_fetch_full_post(base, slug, {})`, which
works cookie-less for PUBLIC posts. The mocks mirror that (archive posts carry no body).
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import ingest_substack as fp
from pipeline.kb import ingest_curation as ic
from pipeline.kb import resolve, schema


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
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "_fetch_all_posts", lambda base, since=None: [dict(p) for p in posts])

    def _full(base, slug, cookies):
        b = _BODY if bodies is None else bodies.get(slug)
        if b is BLOCKED:
            raise sub.SubstackFetchError(f"cloudflare challenge for {slug!r}")
        return {"body_html": b} if b else None
    monkeypatch.setattr(sub, "_fetch_full_post", _full)


# ── a STOPPED archive listing ingests nothing (fail-safe on an incomplete inventory) ──

def test_blocked_listing_writes_no_atoms_and_reports_undetermined(conn, fake_embedder,
                                                                  monkeypatch):
    """A partial listing must not be ingested as a whole archive. We cannot tell "this author
    has N posts" from "we were stopped after N", and the unbounded pull would record the
    truncation as the finished footprint with nothing to re-run it."""
    from pipeline.ingestion.sources import substack as sub

    def _boom(base, since=None):
        raise sub.SubstackListingError("archive listing stopped at offset=50")
    monkeypatch.setattr(sub, "_fetch_all_posts", _boom)

    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB,
                                     handle="carol", author_name="Carol")
    assert out["added"] == 0
    assert out["undetermined"] == 1          # readable signal, not silence
    assert "listing incomplete" in out["error"]
    assert conn.execute("SELECT COUNT(*) c FROM atoms").fetchone()["c"] == 0


def test_blocked_listing_leaves_nothing_marked_seen(conn, fake_embedder, monkeypatch):
    """Nothing is marked `seen`, so the NEXT run redoes the whole walk — the property that
    makes a block retryable rather than a permanent hole in the footprint."""
    from pipeline.ingestion.sources import substack as sub

    calls = {"n": 0}

    def _flaky(base, since=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sub.SubstackListingError("archive listing stopped at offset=0")
        return [dict(_post(99))]
    monkeypatch.setattr(sub, "_fetch_all_posts", _flaky)
    monkeypatch.setattr(sub, "_fetch_full_post", lambda base, slug, cookies: {"body_html": _BODY})

    first = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB,
                                       handle="carol", author_name="Carol")
    assert first["added"] == 0
    second = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB,
                                        handle="carol", author_name="Carol")
    assert second["added"] == 1              # the block cleared and the post arrived
    assert conn.execute("SELECT COUNT(*) c FROM atoms").fetchone()["c"] == 1


# ── the happy path: one public post → one full-body footprint atom ────────────────

def test_footprint_builds_full_body_atom(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [_post(99)])
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB,
                                     handle="carol", author_name="Carol")
    assert out["added"] == 1 and out["paywalled"] == 0 and out["no_body"] == 0

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id='substack:99'").fetchone()
    assert atom["source_type"] == "substack"
    assert atom["entry_mode"] == "oracle-footprint"      # NOT user-saved
    assert atom["who_id"] == "substack:carol"
    assert atom["what_kind"] == "opinion"
    assert json.loads(atom["payload"]) == {"word_count": 900,
                                           "body_state": "complete",
                                           "body_basis": "stated"}

    # Body is chunked clean (no YAML/HTML chrome); the outbound ref becomes an edge.
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='substack:99'").fetchall())
    assert "Autonomous agents compose small tools" in body
    assert "source: substack" not in body and "<p>" not in body
    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id='substack:99'").fetchone()["who_id"]
    assert who == "substack:carol"


# ── D: VLM describes inline images → the description becomes searchable chunk text ──

def test_footprint_vlm_describes_inline_images(conn, fake_embedder, monkeypatch):
    img = "https://substackcdn.com/image/fetch/$s_!abc/https%3A//example.com/chart.png"
    body = f'<p>Here is the data.</p><img src="{img}" alt="chart"><p>As you can see.</p>'
    _patch(monkeypatch, [_post(7, slug="chart-post")], bodies={"chart-post": body})
    # Mock the image read (no live call); the shared cache lives under the sandbox OPYT_HOME.
    # Since 2026-08-02 the long-form path reads through the OCR cascade, not describe_image.
    from pipeline import ocr_cascade
    monkeypatch.setattr(ocr_cascade, "read_image",
                        lambda url, context="": ocr_cascade.MediaRead(
                            "a bar chart of model benchmark scores", "chart", True))

    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    assert out["added"] == 1
    body_txt = " ".join(r["text"] for r in
                        conn.execute("SELECT text FROM chunks WHERE atom_id='substack:7'"))
    # The image's meaning is now searchable body text (the bare CDN link alone carried nothing).
    assert "*Image:* a bar chart of model benchmark scores" in body_txt


# ── policy B: the second run skips already-ingested posts BEFORE any re-fetch ──────

def test_footprint_idempotent_policy_b(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [_post(99)])
    fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")

    # Second run: the per-post body fetch must NOT be called again (policy B skips before it).
    from pipeline.ingestion.sources import substack as sub

    def _boom(*a, **k):
        raise AssertionError("full-body re-fetched for an already-ingested footprint post")

    monkeypatch.setattr(sub, "_fetch_full_post", _boom)
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    assert out["added"] == 0 and out["skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='substack:99'").fetchone()[0] == 1


# ── paywalled + no-body are SKIPPED and counted (never a partial atom) ─────────────

def test_footprint_skips_paywalled_and_bodyless(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [
        _post(1, audience="only_paid", slug="paid"),   # paywalled → skip BEFORE the fetch
        _post(2, slug="nobody"),                        # per-post fetch yields no body → skip
        _post(3, slug="good"),                          # good → ingested
    ], bodies={"paid": _BODY, "nobody": None, "good": _BODY})
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    # `stage_seconds`/`stage_latency` carry wall-clock, so they are dropped before the exact
    # comparison — asserting them would make this test flaky by construction. Their PRESENCE is
    # still checked, since the Step-3 decision depends on the instrumentation actually reporting.
    # `llm_upstreams` joins them: same wall-clock problem, and it is keyed by whichever upstream
    # OpenRouter happened to route to, which is not a property of this test at all.
    _DIAG = ("stage_seconds", "stage_latency", "llm_call_latency", "llm_upstreams")
    assert set(out) >= set(_DIAG)
    counts = {k: v for k, v in out.items() if k not in _DIAG}
    # `dispatched` counts posts handed to the producer pool: the paywalled one is skipped before the
    # fetch and the body-less one after it, so only the good post is dispatched. `producer_failed`
    # is the gap between dispatch and what the consumer saw — a producer that RAISED. It must be 0
    # here, and a non-zero value is the ONLY place such a post appears in the summary at all.
    assert counts == {"source": "substack-footprint", "added": 1, "skipped": 0,
                      "paywalled": 1, "no_body": 1, "undetermined": 0, "failed": 0,
                      "dispatched": 1, "producer_failed": 0,
                      "gate_rejected": 0, "total": 1}
    assert out["stage_latency"]["body_fetch"]["count"] == 2, \
        "the per-post fetch is timed once per post that reached it (paywalled skips before it)"
    ids = {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")}
    assert ids == {"substack:3"}


# ── a BLOCKED fetch is counted apart from a body-less post, and stays retryable ────
# The whole point of the three-verdict split: "Cloudflare stopped us" and "this post is a
# podcast" used to be the same `None`, so a throttled run reported itself as a thin archive.

def test_footprint_block_counts_undetermined_not_no_body(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [
        _post(1, slug="blocked"),      # fetch RAISES → undetermined (we learned nothing)
        _post(2, slug="nobody"),       # fetch returns no body → no_body (confirmed empty)
        _post(3, slug="good"),
    ], bodies={"blocked": BLOCKED, "nobody": None, "good": _BODY})
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")

    assert out["undetermined"] == 1, "a challenge must not be folded into no_body"
    assert out["no_body"] == 1, "a genuinely body-less post is still its own verdict"
    assert out["added"] == 1
    ids = {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")}
    assert ids == {"substack:3"}, "neither skip may write an atom"


def test_footprint_blocked_post_is_retried_next_run(conn, fake_embedder, monkeypatch):
    """A block must NOT mark the post processed — the fail-safe contract. Second run, with the
    block cleared, the same post ingests. (This is what separates it from GitHub's seam, which
    wrote a degraded atom and marked it seen.)"""
    _patch(monkeypatch, [_post(7, slug="flaky")], bodies={"flaky": BLOCKED})
    first = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    assert first["undetermined"] == 1 and first["added"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0

    _patch(monkeypatch, [_post(7, slug="flaky")], bodies={"flaky": _BODY})   # block clears
    second = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    assert second["added"] == 1 and second["skipped"] == 0, "a block must leave no `seen` mark"
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='substack:7'").fetchone()[0] == 1


# ── the `limit` bound caps NEW atoms ──────────────────────────────────────────────

def test_footprint_limit_caps_new_atoms(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [_post(1, slug="a"), _post(2, slug="b"), _post(3, slug="c")])
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB,
                                     handle="carol", limit=2)
    assert out["added"] == 2
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2


# ── ARC-1 Job A: batch the embed across posts (the long-document win) ──────────────

def test_footprint_batches_embed_across_posts(conn, recording_embedder, monkeypatch):
    # Four public posts pool into ONE flush = ONE embed call under the batching sink (per-post
    # store_atom paid four). `added` is a durable-write count (fires at flush), and the `limit`
    # cap still holds because it counts synchronous submits, not the lagging `added`.
    _patch(monkeypatch, [_post(i, slug=f"p{i}") for i in range(1, 5)])
    out = fp.sync_substack_footprint(conn, recording_embedder, publication_url=_PUB, handle="carol")
    assert out["added"] == 4 and out["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 4
    assert len(recording_embedder.calls) == 1                      # four posts, ONE flush


# ── the shared key: a saved post + the SAME footprint post = ONE atom ─────────────

def test_footprint_collides_with_curation_save_no_duplicate(conn, fake_embedder, monkeypatch):
    # First: the user SAVED post 99 (curation, entry_mode=user-saved).
    from pipeline.ingestion.sources import substack as sub
    rec = {"id": 99, "title": "Scaling Laws", "subtitle": "", "post_date": "2026-05-01T00:00:00Z",
           "author_handle": "carol", "author_name": "Carol", "publication_name": "Carol Writes",
           "publication_url": _PUB, "wordcount": 900, "audience": "everyone", "slug": "scaling-laws",
           "url": f"{_PUB}/p/scaling-laws", "preview": "preview"}
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda cookies: [rec])
    monkeypatch.setattr(sub, "_fetch_full_post", lambda base, slug, cookies: {"body_html": _BODY})
    monkeypatch.setattr(sub, "_is_paywalled", lambda r: False)
    ic.sync_substack_saved(conn, fake_embedder)
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='substack:99'"
                        ).fetchone()["entry_mode"] == "user-saved"

    # Then: post 99 shows up again as the Oracle's footprint → policy B skips it, no duplicate,
    # and the user-saved provenance is NOT clobbered.
    _patch(monkeypatch, [_post(99)])
    out = fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    assert out["added"] == 0 and out["skipped"] == 1
    row = conn.execute("SELECT COUNT(*) c, MAX(entry_mode) em FROM atoms WHERE atom_id='substack:99'"
                      ).fetchone()
    assert row["c"] == 1 and row["em"] == "user-saved"


# ── attribution: resolve unifies the footprint author into the Oracle's canonical ─

def test_footprint_author_resolves_to_oracle_canonical(conn, fake_embedder, monkeypatch):
    # An X Oracle whose website field attests the substack home (the merge edge).
    schema.upsert_entity(conn, "x:user:5", name="Carol", identity_links=[_PUB])
    _patch(monkeypatch, [_post(99)])
    fp.sync_substack_footprint(conn, fake_embedder, publication_url=_PUB, handle="carol")
    resolve.resolve_entities(conn)

    x_cid = schema.get_entity(conn, "x:user:5")["canonical_id"]
    sub_cid = schema.get_entity(conn, "substack:carol")["canonical_id"]
    assert x_cid == sub_cid          # one canonical → the footprint attributes to the Oracle


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
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "_fetch_full_post", lambda base, slug, cookies: full)


def test_atom_from_url_mints_referenced_post(conn, fake_embedder, monkeypatch):
    _patch_post(monkeypatch, _full_post(700))
    atom_id = fp.substack_atom_from_url(conn, fake_embedder, "https://joe.substack.com/p/essay")
    assert atom_id == "substack:700"
    row = conn.execute("SELECT who_id, what_kind, entry_mode, source_type FROM atoms "
                       "WHERE atom_id='substack:700'").fetchone()
    assert row["who_id"] == "substack:joe" and row["what_kind"] == "opinion"
    assert row["entry_mode"] == "author_referenced" and row["source_type"] == "substack"


def test_atom_from_url_not_a_post(conn, fake_embedder):
    # A profile / about page (no /p/slug) is not a post → None (nothing to mint or vouch).
    assert fp.substack_atom_from_url(conn, fake_embedder, "https://joe.substack.com/about") is None


def test_atom_from_url_paywalled_skipped(conn, fake_embedder, monkeypatch):
    _patch_post(monkeypatch, _full_post(701, audience="only_paid"))   # preview-only → never stored
    assert fp.substack_atom_from_url(conn, fake_embedder, "https://joe.substack.com/p/paid") is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_atom_from_url_policy_b_never_clobbers(conn, fake_embedder, monkeypatch):
    # A post already present (e.g. from a curation SAVE) must NOT be re-rendered/overwritten — the
    # referenced path returns its id for the vouch and leaves the existing provenance intact.
    schema.upsert_atom(conn, {"atom_id": "substack:702", "source_type": "substack",
                              "entry_mode": "user-saved", "raw_hash": "seed"})
    _patch_post(monkeypatch, _full_post(702, slug="dup"))
    atom_id = fp.substack_atom_from_url(conn, fake_embedder, "https://joe.substack.com/p/dup")
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


def test_atom_from_url_reader_url_resolved_then_minted(conn, fake_embedder, monkeypatch):
    # A reader-url resolves to an a16zcrypto post → minted under THAT publication's who_id, proving
    # the canonical pub (not the reader host `substack.com`) is what attribution keys on.
    monkeypatch.setattr(fp, "_resolve_reader_url",
                        lambda u: "https://a16zcrypto.substack.com/p/agents-real")
    _patch_post(monkeypatch, _full_post(800, slug="agents-real",
                                        host="https://a16zcrypto.substack.com"))
    atom_id = fp.substack_atom_from_url(conn, fake_embedder,
                                        "https://substack.com/home/post/p-194563373")
    assert atom_id == "substack:800"
    assert conn.execute("SELECT who_id FROM atoms WHERE atom_id='substack:800'"
                        ).fetchone()["who_id"] == "substack:a16zcrypto"
