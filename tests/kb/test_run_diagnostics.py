"""Per-stage timing has to REACH the report, not merely be produced.

Every content adapter already built a `StageTimer` and emitted `stage_seconds` / `stage_latency`.
`run_stats` then dropped all of it, because it copied keys only `if isinstance(v, int)` — so every
dict was filtered out by construction on the way to the caller. The numbers survived one place
only: stringified inside the `detail` field, which is how the rerun9 baseline ended up regexing a
latency table back out of a debug string.

What these pin:
  • dict-valued diagnostics travel through `run_stats` into `results[].stats`, while the
    user-facing integer counters keep their own filter (the two sets stay separate);
  • the router timings that exist nowhere else — `curation_pull`'s six sources and
    `onboard_footprint`'s per-source-type clock, which is the ONLY clock `sync_github` has.
"""
from __future__ import annotations

from pipeline.kb import ingest_common, onboard_footprint, schema


def test_run_stats_carries_dict_diagnostics_and_int_counters(kb_home):
    summary = {
        "added": 4, "dispatched": 9,                       # user-facing counters
        "stage_seconds": {"fetch": 1.5}, "stage_latency": {"fetch": {"p95": 2.0}},
        "llm_call_latency": {"vision": {"count": 3}}, "llm_upstreams": {"Nebius": 3},
        "source": "blog", "total": 12,                     # neither set — must NOT be copied
    }
    out = ingest_common.run_stats(summary)
    assert out["added"] == 4 and out["dispatched"] == 9
    assert out["stage_latency"] == {"fetch": {"p95": 2.0}}
    assert out["stage_seconds"] == {"fetch": 1.5}
    assert out["llm_call_latency"] == {"vision": {"count": 3}}
    assert out["llm_upstreams"] == {"Nebius": 3}
    assert "source" not in out and "total" not in out


def test_the_two_key_sets_stay_disjoint():
    """`RUN_STAT_KEYS` is what a USER is told; `RUN_DIAG_KEYS` is what an ENGINEER reads. One key
    in both would mean a counter can't be stripped from a user-facing payload without also
    stripping a diagnostic (or the reverse)."""
    assert not set(ingest_common.RUN_STAT_KEYS) & set(ingest_common.RUN_DIAG_KEYS)


def test_a_wrongly_shaped_diagnostic_is_dropped_not_forwarded(kb_home):
    """Each set has its OWN type filter, so a key carrying the wrong shape is dropped here rather
    than handed on to break whoever reads the report."""
    out = ingest_common.run_stats({"stage_latency": "0.4s", "added": {"n": 1}})
    assert out == {}


def test_onboard_footprint_reports_a_per_source_clock(kb_home, monkeypatch, fake_embedder):
    """The router-level clock: `sync_github` carries no StageTimer of its own, so without this it
    contributes nothing to a wall-clock profile except an unexplained gap. The eligibility gate is
    likewise timed nowhere else."""
    from pipeline.kb import eligibility, ingest_blog, ingest_github

    monkeypatch.setattr(ingest_github, "sync_github",
                        lambda conn, emb, **kw: {"added": 3, "stage_seconds": {}})
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, emb, **kw: {"added": 2,
                                                 "stage_latency": {"fetch": {"p95": 1.0}}})
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("ingest", "stub"))

    conn = schema.connect()
    out = onboard_footprint.onboard_footprint(
        conn, fake_embedder, "x:user:1",
        [{"source_type": "blog", "url": "https://nia.dev", "metadata": {},
          "trust": {"trusted": True}},
         {"source_type": "github", "url": "https://github.com/nia", "metadata": {"handle": "nia"},
          "trust": {"trusted": True}}])

    assert {"blog", "github", "eligibility_gate"} <= set(out["stage_seconds"])
    assert set(out["stage_latency"]["github"]) == {"count", "mean", "p50", "p95", "max"}
    # …and the adapters' own diagnostics survived the trip through run_stats onto the records.
    blog_rec = next(r for r in out["results"] if r["type"] == "blog")
    assert blog_rec["stats"]["stage_latency"] == {"fetch": {"p95": 1.0}}
    conn.close()
