"""Stage-5 footprint expansion (expand.py) — routing + partition, fully offline.

Proves the load-bearing behavior with a FAKE `discover_fn` and STUBBED adapters (no network,
no embeds): trusted sources route to their adapter, needs-review sources are RETURNED not
ingested (Pick #2), non-routable types are skipped-not-dropped, the @handle is read from
`profile.handle`, a handle-less Oracle fails safe, and the cluster is seeded as a trust root.
The real discovery + adapter behavior is proven by their own tests + the live Stage-5 run.
"""
from __future__ import annotations

import pytest

from pipeline.kb import eligibility, expand, oracles, resolve, schema


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


def _fake_discover(sources):
    # discover_profile is now called as discover_fn(seed, seed_type=...) — accept both.
    return lambda handle, seed_type="x": {"username": handle, "sources": sources}


def _src(stype, url, trusted, reasons=None):
    return {"source_type": stype, "url": url,
            "trust": {"trusted": trusted, "reasons": reasons or []}}


@pytest.fixture()
def stub_adapters(monkeypatch):
    """Replace the three atom-KB ingesters with no-op recorders so routing is testable offline.
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
    monkeypatch.setattr(expand.ingest_x_footprint, "sync_x_footprint", mk("x-footprint"))
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("ingest", "stub-eligible"))
    return calls


def test_trusted_routed_review_held_nonroutable_skipped(conn, stub_adapters):
    o = _oracle_person(conn)
    srcs = [
        _src("substack", "https://carol.substack.com", True, ["X-attested (root)"]),
        _src("blog", "https://carol.dev", False, ["no corroboration"]),   # JS/no back-edge → review
        _src("github", "https://github.com/carol", True, ["Bidirectional with an X-attested root"]),
        _src("youtube", "https://youtube.com/@carol", True),               # trusted but no adapter
    ]
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(srcs))

    assert r["handle"] == "carol"
    # substack + github route; the X ROOT footprint always pulls (handle present) → "x".
    assert {i["source_type"] for i in r["ingested"]} == {"substack", "github", "x"}
    assert [nr["source_type"] for nr in r["needs_review"]] == ["blog"]
    assert r["needs_review"][0]["reason"] == "no corroboration"
    assert [sk["source_type"] for sk in r["skipped"]] == ["youtube"]
    # the adapters were actually invoked for the trusted routable sources + the X root — not the review one
    assert {c[0] for c in stub_adapters} == {"substack", "github", "x-footprint"}


def test_needs_review_is_never_ingested(conn, stub_adapters):
    """A trusted-looking type that's NOT trust-verified must not touch an OFF-X adapter. (The X
    ROOT always pulls — it's not a discovered source, so it isn't subject to the trust gate.)"""
    o = _oracle_person(conn)
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("substack", "https://squatter.substack.com", False, ["no trust path"])]))
    assert r["needs_review"][0]["url"] == "https://squatter.substack.com"
    assert [i["source_type"] for i in r["ingested"]] == ["x"]   # squatter NOT ingested; only X root
    assert {c[0] for c in stub_adapters} == {"x-footprint"}     # off-X adapters untouched


def test_x_root_footprint_always_pulled(conn, stub_adapters):
    """X is the ROOT, not a discovered source: the timeline pull fires on the handle alone, even
    when discovery surfaces NO off-X sources."""
    o = _oracle_person(conn)
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover([]))
    assert {i["source_type"] for i in r["ingested"]} == {"x"}
    xf = [c for c in stub_adapters if c[0] == "x-footprint"]
    assert len(xf) == 1 and xf[0][1]["handle"] == "carol"


def test_discovery_crash_still_pulls_x_root(conn, stub_adapters):
    """DECOUPLED: an off-X discovery crash records `discovery_error` but must NOT skip the X root
    (the primary channel). No early return — the timeline still pulls, trust roots still seed."""
    def _boom(handle, seed_type="x"):
        raise RuntimeError("web-search API down")
    o = _oracle_person(conn)
    r = expand.expand_oracle(conn, None, o, discover_fn=_boom)
    assert "discovery_failed" in r["discovery_error"]
    assert {i["source_type"] for i in r["ingested"]} == {"x"}   # X pulled despite the discovery crash


def test_github_owner_parsed_from_url(conn, stub_adapters):
    o = _oracle_person(conn)
    expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("github", "https://github.com/carolcorp", True, ["X-attested"])]))
    gh = [c for c in stub_adapters if c[0] == "github"][0]
    assert gh[1]["handles"] == ["carolcorp"]


def test_no_rootable_profile_is_failsafe(conn, stub_adapters):
    # An X person with no handle AND no Substack member → nothing to root discovery on.
    schema.upsert_entity(conn, "x:user:2", name="NoHandle")  # no profile.handle
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:2"])
    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:2"][0]
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover([_src("blog", "x", True)]))
    assert r["error"] == "no_rootable_profile"
    assert r["ingested"] == [] and stub_adapters == []


def _oracle_substack(conn, eid="substack:carol", *, name="Carol"):
    """Upsert a Substack-ONLY person (no x:user member), resolve, confirm as Oracle."""
    schema.upsert_entity(conn, eid, name=name, identity_links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[eid])
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def test_substack_root_routes_and_pulls_discovered_x(conn, stub_adapters):
    """De-X-rooting: a Substack-ONLY Oracle roots on their Substack (no X member). Discovery
    surfaces their trusted Substack (routed via Half-A) + their trusted X (pulled as a timeline
    via Half-B — the Oracle's X TL is pulled whenever found, regardless of root platform); the
    X link itself is NOT routed as a source."""
    o = _oracle_substack(conn)
    srcs = [
        _src("substack", "https://carol.substack.com", True, ["X-attested (root)"]),
        _src("x", "https://x.com/carolx", True, ["Identity-attested by trusted them.substack.com"]),
    ]
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(srcs))

    assert r["root"] == {"seed": "carol", "seed_type": "substack"}
    assert r.get("error") is None
    # substack routed (Half-A) + the discovered X pulled as a timeline (Half-B).
    assert {i["source_type"] for i in r["ingested"]} == {"substack", "x"}
    xf = [c for c in stub_adapters if c[0] == "x-footprint"]
    assert len(xf) == 1 and xf[0][1]["handle"] == "carolx"     # discovered X handle, not the root
    assert {c[0] for c in stub_adapters} == {"substack", "x-footprint"}


def test_substack_root_with_no_x_skips_the_timeline(conn, stub_adapters):
    """A Substack-only Oracle whose discovery finds NO X → the X-footprint pull is skipped
    gracefully (fail-safe); only their Substack archive (Half-A) ingests."""
    o = _oracle_substack(conn, eid="substack:dave", name="Dave")
    srcs = [_src("substack", "https://dave.substack.com", True, ["X-attested (root)"])]
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(srcs))

    assert r["root"] == {"seed": "dave", "seed_type": "substack"}
    assert {i["source_type"] for i in r["ingested"]} == {"substack"}   # no X pulled
    assert {c[0] for c in stub_adapters} == {"substack"}               # x-footprint NOT invoked




def test_oracle_read_survives_canonical_shift(conn, stub_adapters):
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


def test_eligibility_skip_blocks_website_adapter(conn, stub_adapters, monkeypatch):
    """The load-bearing fix: a multi-author site the gate SKIPS must never reach the website
    adapter — the trust-laundering the footprint-adapter guard exists to stop. The X root still
    pulls (it's ungated by construction); the substack adapter is never invoked."""
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("skip", "multi-author/org site"))
    o = _oracle_person(conn)
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("substack", "https://team.substack.com", True, ["X-attested"])]))

    assert {c[0] for c in stub_adapters} == {"x-footprint"}          # substack adapter NEVER ran
    sub = [i for i in r["ingested"] if i["source_type"] == "substack"][0]
    assert sub["skipped"] == "eligibility:skip" and "multi-author" in sub["reason"]


def test_github_is_not_gated(conn, stub_adapters, monkeypatch):
    """GitHub attributes to the ATTESTED repo owner, not the Oracle → no inference to launder →
    the eligibility gate must NOT be consulted for it (only website adapters are gated). Locks the
    attested-vs-inferred rule the guard's adapter-name scope encodes."""
    gate_urls = []

    def spy(conn, url, **kw):
        gate_urls.append(url)
        return eligibility.GateDecision("ingest", "spy")

    monkeypatch.setattr(eligibility, "gate", spy)
    o = _oracle_person(conn)
    expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("github", "https://github.com/carol", True, ["X-attested"])]))

    assert gate_urls == []                                          # github never touched the gate
    assert "github" in {c[0] for c in stub_adapters}               # …but the repo adapter still ran


def test_website_gate_consulted_with_oracle_name(conn, stub_adapters, monkeypatch):
    """The website gate runs BEFORE the adapter and is passed the Oracle's name as
    `expected_author` — that's what arms the 'single-authored, but by someone ELSE' squatter check."""
    seen = []

    def spy(conn, url, *, expected_author=None, **kw):
        seen.append((url, expected_author))
        return eligibility.GateDecision("ingest", "spy")

    monkeypatch.setattr(eligibility, "gate", spy)
    o = _oracle_person(conn)                                        # name="Carol"
    expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("blog", "https://carol.dev", True, ["X-attested"])]))

    assert seen == [("https://carol.dev", "Carol")]


# ── Lookback selectors (onboarding: X window + Substack/blog window) ───────────

def test_lookback_presets_match_spec():
    # X: 6mo/1yr/2yr (a hard 2yr ceiling). Substack/blog: 1yr/2yr/5yr/all ('all' = no bound).
    assert expand.X_LOOKBACK_PRESETS == {"6mo": 183, "1yr": 365, "2yr": 730}
    assert expand.WEB_LOOKBACK_PRESETS == {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
    assert expand._since_from_days(None) is None                   # 'all' → no lower bound
    assert expand._since_from_days(365) is not None


def test_lookback_windows_thread_to_adapters(conn, stub_adapters):
    # The two selectors reach the right adapters: X window → sync_x_footprint, Substack/blog
    # window → sync_substack_footprint. (Presets resolve to `since` datetimes at the CLI; here
    # we pass the datetimes directly.)
    from datetime import datetime, timezone
    x_since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    web_since = datetime(2024, 6, 1, tzinfo=timezone.utc)
    o = _oracle_person(conn)
    expand.expand_oracle(
        conn, None, o,
        discover_fn=_fake_discover([_src("substack", "https://carol.substack.com", True, ["root"])]),
        x_since=x_since, web_since=web_since,
    )
    by_name = {name: kw for name, kw in stub_adapters}
    assert by_name["x-footprint"]["since"] == x_since
    assert by_name["substack"]["since"] == web_since


def test_lookback_defaults_none_preserves_prior_behavior(conn, stub_adapters):
    # No selector → since=None reaches both adapters (X then falls to its own 6mo default,
    # Substack to full archive) — the pre-selector behavior is unchanged.
    o = _oracle_person(conn)
    expand.expand_oracle(
        conn, None, o,
        discover_fn=_fake_discover([_src("substack", "https://carol.substack.com", True, ["root"])]))
    by_name = {name: kw for name, kw in stub_adapters}
    assert by_name["x-footprint"]["since"] is None
    assert by_name["substack"]["since"] is None


# ── Blog rooting (a blog-only Oracle roots on their blog) ──────────────────────

def _oracle_blog(conn, eid="blog:carol.dev", *, name="Carol", url="https://carol.dev"):
    """A blog-ONLY person (no x:user / substack member), resolved + confirmed as an Oracle."""
    schema.upsert_entity(conn, eid, name=name, identity_links=[url])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[eid])
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def test_first_url_reads_json_string_or_list():
    assert expand._first_url('["https://a.dev", "x"]') == "https://a.dev"
    assert expand._first_url(["https://b.dev"]) == "https://b.dev"
    assert expand._first_url(None) is None
    assert expand._first_url("not-a-url") is None


def test_blog_home_reconstructed_from_id_when_no_links():
    o = {"members": [{"entity_id": "blog:simonwillison.net", "identity_links": None}]}
    assert expand._blog_home(o) == "https://simonwillison.net"


def test_blog_only_oracle_roots_on_its_blog(conn, stub_adapters):
    """De-X-rooting for blogs: a blog-only Oracle roots on `{seed: home, seed_type: 'blog'}` and
    its blog routes through the blog footprint adapter. No X exists, so no timeline is pulled."""
    o = _oracle_blog(conn)
    r = expand.expand_oracle(conn, None, o, discover_fn=_fake_discover(
        [_src("blog", "https://carol.dev", True, ["root (self)"])]))

    assert r["root"] == {"seed": "https://carol.dev", "seed_type": "blog"}
    assert r.get("error") is None
    assert {i["source_type"] for i in r["ingested"]} == {"blog"}       # blog routed, no X pull
    assert {c[0] for c in stub_adapters} == {"blog"}
