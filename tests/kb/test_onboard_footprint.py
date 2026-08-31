"""onboard_footprint — the atom-KB router for a confirmed Oracle's discovered profiles.

Each source routes exactly once: personal blog/substack through the single-author
gate to the footprint adapter (gate-skip → affiliation), github to sync_github with
resolve, org-shaped links to an affiliation, scholar/orcid deferred. The
footprint adapters + sync_github are monkeypatched (no network / no embed); the
affiliation path uses the REAL record_affiliation (mirrors test_affiliation).
"""

from __future__ import annotations

import pytest

from pipeline.kb import eligibility, ingest_blog, ingest_github, ingest_substack, resolve, schema
from pipeline.kb.eligibility import AuthorshipVerdict, GateDecision
from pipeline.kb.onboard_footprint import onboard_footprint


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    schema.upsert_entity(c, "x:user:7", name="Simon")   # the Oracle
    yield c
    c.close()


def _src(source_type, url, *, trusted=True, shape="personal", handle=None):
    meta = {"shape": shape}
    if handle:
        meta["handle"] = handle
    return {"source_type": source_type, "url": url, "metadata": meta,
            "trust": {"trusted": trusted}}


# ── org-shaped surfaced link → affiliation (no atoms) ─────────────────────────

def test_org_shaped_link_records_affiliation(conn):
    src = _src("github", "https://github.com/orgs/anthropics", trusted=False, shape="org")
    out = onboard_footprint(conn, None, "x:user:7", [src], author_name="Anthropic")

    assert out["affiliations"] == 1 and out["ingested"] == 0
    # The org node is kept so the SKIP is not a silent discard. The person→org relation rode an
    # `affiliated_with` edge until that table was deleted 2026-08-23 with no reader.
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id LIKE 'org:%'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0    # a fact, not content


# ── personal blog → gate → footprint adapter ──────────────────────────────────

def test_personal_blog_gated_then_ingested(conn, monkeypatch):
    calls = {}
    monkeypatch.setattr(eligibility, "gate",
                        lambda *a, **k: GateDecision("ingest", "single-author, eligible",
                                                     AuthorshipVerdict("single", author_name="Simon")))
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, emb, **kw: calls.update(kw) or {"added": 3})
    monkeypatch.setattr(resolve, "resolve_entities", lambda conn: None)

    src = _src("blog", "https://simonwillison.net", handle="simonwillison")
    out = onboard_footprint(conn, object(), "x:user:7", [src], author_name="Simon")

    assert out["ingested"] == 1
    assert calls["blog_url"] == "https://simonwillison.net"     # adapter really ran, post-gate


def test_personal_substack_gated_then_ingested(conn, monkeypatch):
    calls = {}
    monkeypatch.setattr(eligibility, "gate",
                        lambda *a, **k: GateDecision("ingest", "single", AuthorshipVerdict("single")))
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda conn, emb, **kw: calls.update(kw) or {"added": 5})
    monkeypatch.setattr(resolve, "resolve_entities", lambda conn: None)

    src = _src("substack", "https://carol.substack.com", handle="carol")
    out = onboard_footprint(conn, object(), "x:user:7", [src], author_name="Carol")

    assert out["ingested"] == 1
    assert calls["publication_url"] == "https://carol.substack.com" and calls["handle"] == "carol"


# ── github → sync_github → resolve unifies the owner into the Oracle ──────────
#
# It used to also pass `same_entity={login: oracle_id}` to attest the github↔x link a second time
# as an edge. `resolve.py` reads only `identity_links`, so that edge was redundant even before the
# `edges` table was deleted 2026-08-23 — which is why the parameter went with it.

def test_github_profile_ingests_and_resolves(conn, monkeypatch):
    calls = {}
    monkeypatch.setattr(ingest_github, "sync_github",
                        lambda conn, emb, **kw: calls.update(kw) or {"added": 2})
    monkeypatch.setattr(resolve, "resolve_entities", lambda conn: None)

    src = _src("github", "https://github.com/simonw", handle="simonw")
    out = onboard_footprint(conn, object(), "x:user:7", [src])

    assert out["ingested"] == 1
    assert calls["handles"] == ["simonw"]


# ── gate SKIP (multi-author) → affiliation, no atoms ──────────────────────────

def test_blog_gate_skip_becomes_affiliation(conn, monkeypatch):
    monkeypatch.setattr(eligibility, "gate",
                        lambda *a, **k: GateDecision("skip", "multi-author/org site",
                                                     AuthorshipVerdict("multi")))
    # NB: sync_blog_footprint is NOT patched — a skip must never reach it (guard invariant).
    src = _src("blog", "https://anthropic.com", handle="anthropic")
    out = onboard_footprint(conn, object(), "x:user:7", [src], author_name="Anthropic")

    assert out["affiliations"] == 1 and out["ingested"] == 0
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id LIKE 'org:%'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── boundaries: untrusted skipped, gate needs-review parked, scholar deferred ─

def test_untrusted_non_org_is_parked_not_ingested(conn, monkeypatch):
    # If gate were reached it would raise (proving we never got there for an untrusted src).
    monkeypatch.setattr(eligibility, "gate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate reached")))
    src = _src("blog", "https://random.dev", trusted=False, shape="unknown")
    out = onboard_footprint(conn, object(), "x:user:7", [src])
    assert out["needs_review"] == 1 and out["ingested"] == 0


def test_gate_needs_review_parks_the_source(conn, monkeypatch):
    monkeypatch.setattr(eligibility, "gate",
                        lambda *a, **k: GateDecision("needs-review", "could not classify",
                                                     AuthorshipVerdict("unknown")))
    src = _src("blog", "https://unsure.dev")
    out = onboard_footprint(conn, object(), "x:user:7", [src])
    assert out["needs_review"] == 1 and out["ingested"] == 0


def test_scholar_is_deferred(conn):
    src = _src("scholar", "https://scholar.google.com/citations?user=abc", handle="abc")
    out = onboard_footprint(conn, None, "x:user:7", [src])
    assert out["deferred"] == 1


def test_missing_oracle_id_is_an_error(conn):
    out = onboard_footprint(conn, None, "", [_src("github", "https://github.com/x")])
    assert "error" in out
