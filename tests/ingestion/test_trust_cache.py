"""Tests for Batch 3 — trust cache (invalidation) + co-routed host-supplied URLs."""

import importlib

import pytest

dp = importlib.import_module("pipeline.ingestion.discover_profile")  # bypass __init__ shadow
from pipeline.ingestion.trust_types import Edge


# ── Snapshot hash / cache invalidation ───────────────────────────────────────

def test_snapshot_hash_stable_and_order_independent():
    a = dp._x_snapshot_hash("Naval", ["nav.al", "naval.substack.com"])
    b = dp._x_snapshot_hash("Naval", ["naval.substack.com", "nav.al"])  # reordered
    assert a == b


def test_snapshot_hash_changes_on_new_bio_url_or_rename():
    base = dp._x_snapshot_hash("Naval", ["nav.al"])
    assert dp._x_snapshot_hash("Naval", ["nav.al", "x.com/naval"]) != base   # new link
    assert dp._x_snapshot_hash("Naval Ravikant", ["nav.al"]) != base         # rename


class _Cfg:
    """Minimal cfg stub: state_file(name) → a tmp path."""
    def __init__(self, tmp_path):
        self._tmp = tmp_path

    def state_file(self, name):
        return self._tmp / f"{name}.json"


def test_cache_roundtrip_hit_then_miss_on_hash_change(tmp_path):
    cfg = _Cfg(tmp_path)
    result = {"username": "naval", "sources": [{"url": "https://nav.al"}]}
    dp._save_cached_trust("naval", "hash1", result, cfg)

    assert dp._get_cached_trust("naval", "hash1", cfg) == result   # hit
    assert dp._get_cached_trust("naval", "hash2", cfg) is None     # invalidated by hash change
    assert dp._get_cached_trust("other", "hash1", cfg) is None     # different handle


def test_cache_expires_after_ttl(tmp_path, monkeypatch):
    cfg = _Cfg(tmp_path)
    dp._save_cached_trust("naval", "h", {"x": 1}, cfg)
    # Force the stored entry to look ancient.
    import json
    sf = cfg.state_file("trust_cache")
    cache = json.loads(sf.read_text())
    cache["naval"]["evaluated_at"] = "2000-01-01T00:00:00+00:00"
    sf.write_text(json.dumps(cache))
    assert dp._get_cached_trust("naval", "h", cfg) is None         # stale → miss


# ── Co-routed open-web discovery: host FINDS, the graph JUDGES ────────────────

def test_host_supplied_urls_become_typed_candidate_sources():
    """The host's web-search results enter as CANDIDATES, classified by type.

    This is the half of co-routing that was missing. `_web_search_followup` asked the host to
    search and then handed it no coherent way back — it named the vault-era add-a-person tool's
    `include_urls` argument, that tool was deleted 2026-08-07, and `add_oracle` never grew a
    replacement. So the loop was open at the return end.
    """
    got = dp._sources_from_urls([
        "https://alice.dev",
        "https://alice.substack.com",
        "https://www.youtube.com/@alice",
    ])
    by_type = {s.source_type: s for s in got}
    assert set(by_type) == {"blog", "substack", "youtube"}
    # LOW confidence, deliberately: an unverified claim from a chat model is the weakest input
    # this module accepts, and it must not out-rank a deterministic probe on a dedupe collision.
    assert all(s.confidence == "low" for s in got)
    assert all(s.metadata.get("found_by") == "host_web_search" for s in got)


def test_host_supplied_urls_drop_junk_without_raising():
    """FAIL-SAFE: a chat model returns prose, mailto: links and empty strings. None may crash a
    discovery run, and none may enter the graph as a source."""
    got = dp._sources_from_urls(
        ["", "   ", "not a url at all", "mailto:alice@example.com", None,
         "javascript:void(0)", "https://alice.dev"])
    assert [s.url for s in got] == ["https://alice.dev"]


def test_host_supplied_urls_are_UNTRUSTED_until_the_graph_says_otherwise(monkeypatch):
    """⚠️ THE POINT OF ROUTING THEM THROUGH `_compute_trust` AT ALL.

    The host asserting "this is Alice's blog" is a CLAIM, not an attestation. It enters as an
    untrusted candidate and must earn a trust edge exactly like a probe-found source. If this
    ever starts returning trusted, co-routing has become a way to launder a chat model's guess
    into a trust root, which is the one thing the trust graph exists to prevent.
    """
    monkeypatch.setattr(dp, "fetch_landing_edges",
                        lambda source_id, url, relevant, timeout=12: [])
    monkeypatch.setattr(dp, "fetch_profile_links", lambda url: [])
    all_sources = dp._sources_from_urls(["https://alice.dev"])
    verdicts = dp._compute_trust(
        "alice", dp.canonical_identity("x.com/alice"), [],       # NO declared identity links
        {"display_name": "Alice", "bio": "", "website": ""}, [], all_sources,
        skip_edge_fetch=False,
    )
    assert not verdicts[dp.canonical_identity("https://alice.dev")].trusted
