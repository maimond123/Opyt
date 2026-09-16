"""Stage-5 footprint helpers (expand.py) — rooting + routing, fully offline.

`expand` owns no loop: it roots an Oracle for discovery (`_root_profile`, `_x_handle_to_pull`)
and routes ONE discovered source to its adapter (`_route_source`). The orchestration that used
to live here — `expand_oracle` / `expand_all` / the CLI — was deleted 2026-09-04 as a second,
divergent copy of `oracles._ingest_oracle`; these tests therefore exercise the helpers directly
rather than through a driver.

Adapters are STUBBED (no network, no embeds). The real discovery + adapter behavior is proven
by their own tests.
"""
from __future__ import annotations

import pytest

from pipeline.kb import eligibility, expand, link_router, oracles, resolve, schema
from pipeline.kb.onboard_footprint import onboard_footprint


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _oracle_person(conn, eid="x:user:1", *, name="Carol", handle="carol"):
    """Upsert an X person (handle in profile), resolve so canonical_id=self, confirm as Oracle."""
    schema.upsert_entity(conn, eid, name=name, profile={"handle": handle} if handle else None)
    resolve.resolve_entities(conn)          # sets canonical_id (Phase-2 does this in the real flow)
    oracles.confirm(conn, canonical_ids=[eid])
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def _src(stype, url, trusted, reasons=None):
    return {"source_type": stype, "url": url,
            "trust": {"trusted": trusted, "reasons": reasons or []}}


@pytest.fixture()
def stub_adapters(monkeypatch):
    """Replace the atom-KB ingesters with no-op recorders so routing is testable offline.
    Also stubs the eligibility gate to PASS by default — the real gate does a classify (LLM/DB),
    which the offline routing tests must not trigger; the skip-path tests override it."""
    calls = []

    def mk(name):
        def f(conn, embedder, **kw):
            calls.append((name, kw))
            return {"adapter": name}
        return f

    from pipeline.kb import ingest_blog, ingest_substack

    monkeypatch.setattr(ingest_substack, "sync_substack_footprint", mk("substack"))
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint", mk("blog"))
    monkeypatch.setattr(expand.ingest_github, "sync_github", mk("github"))
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("ingest", "stub-eligible"))
    return calls


# ── _route_source: one discovered source → its adapter ────────────────────────

def test_website_source_reaches_its_adapter(conn, stub_adapters):
    r = expand._route_source(conn, None, _src("substack", "https://carol.substack.com", True),
                             author_name="Carol", limit=0)
    assert r["source_type"] == "substack" and "ingested" in r
    assert [c[0] for c in stub_adapters] == ["substack"]


def test_unadapted_type_is_skipped_not_dropped(conn, stub_adapters):
    """scholar/youtube/podcast have no atom-KB adapter. The source must come back RECORDED as
    skipped — a caller that reports it can only report what it is handed."""
    r = expand._route_source(conn, None, _src("youtube", "https://youtube.com/@carol", True),
                             author_name="Carol", limit=0)
    assert r["skipped"] == "no_adapter" and r["url"] == "https://youtube.com/@carol"
    assert stub_adapters == []


def test_github_owner_parsed_from_url(conn, stub_adapters):
    expand._route_source(conn, None, _src("github", "https://github.com/carolcorp", True),
                         author_name="Carol", limit=0)
    gh = [c for c in stub_adapters if c[0] == "github"][0]
    assert gh[1]["handles"] == ["carolcorp"]


def test_a_repo_url_mints_that_repo_rather_than_sweeping_its_owner(conn, stub_adapters,
                                                                   monkeypatch):
    """The refresh rail's half of the 2026-09-05 split. `github.com/acme/memory` is one repository
    the Oracle pointed at; taking its OWNER swept every repo `acme` has ever pushed and never
    atomized the one the bio named. Latent rather than live — this router's only production
    caller synthesizes account-shaped urls — so this test is what keeps it that way."""
    minted = {}
    monkeypatch.setattr(link_router, "mint_artifact",
                        lambda conn, emb, url, kind, **kw: minted.update(url=url, **kw)
                        or {"status": "minted", "atom_id": "github:acme/memory"})

    r = expand._route_source(conn, None, _src("github", "https://github.com/acme/memory", True),
                             author_name="Carol", limit=0)

    assert "ingested" in r and r["ingested"]["added"] == 1
    assert minted["url"] == "https://github.com/acme/memory"
    assert minted["entry_mode"] == "author_referenced"
    assert stub_adapters == []                                      # the account crawl never ran


def test_the_two_rails_route_one_repo_url_the_same_way(conn, stub_adapters, monkeypatch):
    """The reason the split lives in `ingest_github` and not in each router. Both take the same
    shape of source dict, and each deriving the account for itself had already produced two
    different answers for this url — `memory` here, `acme` there. One home, one answer."""
    seen = []
    monkeypatch.setattr(link_router, "mint_artifact",
                        lambda conn, emb, url, kind, **kw: seen.append(url)
                        or {"status": "minted", "atom_id": "github:acme/memory"})
    url = "https://github.com/acme/memory"

    expand._route_source(conn, None, _src("github", url, True), author_name="Carol", limit=0)
    onboard_footprint(conn, None, "x:user:7",
                      [{"source_type": "github", "url": url, "metadata": {"shape": "personal"},
                        "trust": {"trusted": True}}])

    assert seen == [url, url]
    assert stub_adapters == []


def test_github_url_with_no_owner_is_skipped(conn, stub_adapters):
    r = expand._route_source(conn, None, _src("github", "https://github.com", True),
                             author_name="Carol", limit=0)
    assert r["skipped"] == "no_owner_in_url" and stub_adapters == []


def test_eligibility_skip_blocks_website_adapter(conn, stub_adapters, monkeypatch):
    """The load-bearing fix: a multi-author site the gate SKIPS must never reach the website
    adapter — the trust-laundering the footprint-adapter guard exists to stop."""
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("skip", "multi-author/org site"))
    r = expand._route_source(conn, None, _src("substack", "https://team.substack.com", True),
                             author_name="Carol", limit=0)
    assert stub_adapters == []                                      # adapter NEVER ran
    assert r["skipped"] == "eligibility:skip" and "multi-author" in r["reason"]


def test_github_is_not_gated(conn, stub_adapters, monkeypatch):
    """GitHub attributes to the ATTESTED repo owner, not the Oracle → no inference to launder →
    the eligibility gate must NOT be consulted for it (only website adapters are gated)."""
    gate_urls = []

    def spy(conn, url, **kw):
        gate_urls.append(url)
        return eligibility.GateDecision("ingest", "spy")

    monkeypatch.setattr(eligibility, "gate", spy)
    expand._route_source(conn, None, _src("github", "https://github.com/carol", True),
                         author_name="Carol", limit=0)
    assert gate_urls == []                                          # github never touched the gate
    assert "github" in {c[0] for c in stub_adapters}                # …but the repo adapter still ran


def test_website_gate_consulted_with_oracle_name(conn, stub_adapters, monkeypatch):
    """The website gate runs BEFORE the adapter and is passed the Oracle's name as
    `expected_author` — that's what arms the 'single-authored, but by someone ELSE' squatter check."""
    seen = []

    def spy(conn, url, *, expected_author=None, **kw):
        seen.append((url, expected_author))
        return eligibility.GateDecision("ingest", "spy")

    monkeypatch.setattr(eligibility, "gate", spy)
    expand._route_source(conn, None, _src("blog", "https://carol.dev", True),
                         author_name="Carol", limit=0)
    assert seen == [("https://carol.dev", "Carol")]


def test_blocked_adapter_run_is_not_reported_as_ingested(conn, stub_adapters, monkeypatch):
    """Adapters signal a hard stop by RETURNING an error summary, not raising. Without the
    classify step a blocked archive walk (zero atoms, nothing marked seen) reaches the caller
    labelled `ingested`."""
    from pipeline.kb import ingest_blog

    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda *a, **kw: {"error": "403 from the host", "added": 0})
    r = expand._route_source(conn, None, _src("blog", "https://carol.dev", True),
                             author_name="Carol", limit=0)
    assert "ingested" not in r and (r.get("blocked") or r.get("error"))


# ── rooting: which profile discovery starts from ──────────────────────────────

def test_x_member_roots_on_its_handle(conn):
    o = _oracle_person(conn)
    assert expand._root_profile(conn, o) == {"seed": "carol", "seed_type": "x"}


def _oracle_substack(conn, eid="substack:carol", *, name="Carol"):
    """Upsert a Substack-ONLY person (no x:user member), resolve, confirm as Oracle."""
    schema.upsert_entity(conn, eid, name=name, identity_links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[eid])
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def test_substack_only_oracle_roots_on_its_substack(conn, solo_site):
    """De-X-rooting: a Substack-ONLY Oracle roots on that handle, with no X anywhere."""
    o = _oracle_substack(conn)
    assert expand._root_profile(conn, o) == {"seed": "carol", "seed_type": "substack"}


def test_blog_only_oracle_roots_on_its_blog(conn, solo_site):
    schema.upsert_entity(conn, "blog:carol.dev", name="Carol",
                         identity_links=["https://carol.dev"])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["blog:carol.dev"])
    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "blog:carol.dev"][0]
    assert expand._root_profile(conn, o) == {"seed": "https://carol.dev", "seed_type": "blog"}


def test_no_rootable_profile_is_none(conn):
    """An X person with no handle and no Substack/blog member → nothing to root discovery on.
    None, not a crash: the caller reports it."""
    schema.upsert_entity(conn, "x:user:2", name="NoHandle")  # no profile.handle
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:2"])
    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:2"][0]
    assert expand._root_profile(conn, o) is None


def test_oracle_read_survives_canonical_shift(conn):
    """Regression: a footprint merge AFTER confirm shifts the cluster head (blog: sorts below
    x:user:), so the oracle's STORED canonical_id goes stale. confirmed_oracles + _x_handle must
    still recover the full cluster + handle by re-anchoring to the current head."""
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    resolve.resolve_entities(conn)                              # head = x:user:1
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    # simulate the post-footprint-resolve state: blog:carol merged in and became the new head
    schema.upsert_entity(conn, "blog:carol", name="carol")
    conn.execute("UPDATE entities SET canonical_id='blog:carol' "
                 "WHERE entity_id IN ('x:user:1', 'blog:carol')")
    conn.commit()
    o = oracles.confirmed_oracles(conn)[0]
    assert o["canonical_id"] == "blog:carol"                    # reports the CURRENT head
    assert {m["entity_id"] for m in o["members"]} == {"x:user:1", "blog:carol"}
    assert expand._x_handle(conn, "x:user:1") == "carol"        # stale stored id still resolves


# ── _x_handle_to_pull: whose timeline the caller pulls ────────────────────────

def test_x_rooted_oracle_pulls_its_root_handle():
    assert expand._x_handle_to_pull({"seed": "carol", "seed_type": "x"}, {}) == "carol"


def test_substack_rooted_oracle_pulls_a_discovered_trusted_x():
    """An Oracle's X timeline is their richest channel, so it is pulled whenever findable —
    including for a person rooted on another platform."""
    profile = {"sources": [_src("x", "https://x.com/carolx", True, ["Identity-attested"])]}
    root = {"seed": "carol", "seed_type": "substack"}
    assert expand._x_handle_to_pull(root, profile) == "carolx"


def test_an_untrusted_discovered_x_is_not_pulled():
    """Trust is per-source: a squatter's x.com link on a trusted person's page is not their X."""
    profile = {"sources": [_src("x", "https://x.com/squatter", False, ["no trust path"])]}
    root = {"seed": "carol", "seed_type": "substack"}
    assert expand._x_handle_to_pull(root, profile) is None


def test_no_x_anywhere_pulls_nothing():
    root = {"seed": "https://carol.dev", "seed_type": "blog"}
    assert expand._x_handle_to_pull(root, {"sources": []}) is None


# ── Lookback selectors (onboarding: X window + Substack/blog window) ───────────

def test_lookback_presets_match_spec():
    # X: 6mo/1yr/2yr (a hard 2yr ceiling). Substack/blog: 1yr/2yr/5yr/all ('all' = no bound).
    assert expand.X_LOOKBACK_PRESETS == {"6mo": 183, "1yr": 365, "2yr": 730}
    assert expand.WEB_LOOKBACK_PRESETS == {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
    assert expand._since_from_days(None) is None                   # 'all' → no lower bound
    assert expand._since_from_days(365) is not None


def test_web_since_threads_to_the_website_adapter(conn, stub_adapters):
    from datetime import datetime, timezone
    web_since = datetime(2024, 6, 1, tzinfo=timezone.utc)
    expand._route_source(conn, None, _src("substack", "https://carol.substack.com", True),
                         author_name="Carol", limit=0, web_since=web_since)
    assert dict(stub_adapters)["substack"]["since"] == web_since


def test_web_since_defaults_to_none(conn, stub_adapters):
    """No selector → since=None reaches the adapter, which falls to the full archive."""
    expand._route_source(conn, None, _src("blog", "https://carol.dev", True),
                         author_name="Carol", limit=0)
    assert dict(stub_adapters)["blog"]["since"] is None


def test_github_since_and_web_since_are_not_interchangeable(conn, stub_adapters):
    """`web_since` bounds which POSTS to consider; `github_since` skips repos untouched since
    then. A shared `since` would mean a different thing on each adapter."""
    from datetime import datetime, timezone
    gh_since = datetime(2025, 3, 1, tzinfo=timezone.utc)
    expand._route_source(conn, None, _src("github", "https://github.com/carol", True),
                         author_name="Carol", limit=0,
                         web_since=datetime(2020, 1, 1, tzinfo=timezone.utc),
                         github_since=gh_since)
    assert dict(stub_adapters)["github"]["since"] == gh_since


# ── identity_links helpers ────────────────────────────────────────────────────

def test_first_url_reads_json_string_or_list():
    assert expand._first_url('["https://a.dev", "x"]') == "https://a.dev"
    assert expand._first_url(["https://b.dev"]) == "https://b.dev"
    assert expand._first_url(None) is None
    assert expand._first_url("not-a-url") is None


def test_blog_home_reconstructed_from_id_when_no_links():
    o = {"members": [{"entity_id": "blog:simonwillison.net", "identity_links": None}]}
    assert expand._blog_home(o) == "https://simonwillison.net"
