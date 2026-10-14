"""Per-stage timing has to reach the report, not merely be produced.

Every content adapter builds a `StageTimer` and emits `stage_seconds` / `stage_latency`.
`run_stats` must not drop them by copying keys only `if isinstance(v, int)`, which would filter
every dict out on the way to the caller and leave the numbers only stringified inside the
`detail` field.

What these pin:
  • dict-valued diagnostics travel through `run_stats` into `results[].stats`, while the
    user-facing integer counters keep their own filter (the two sets stay separate);
  • the router timings that exist nowhere else — `curation_pull`'s six sources.
"""
from __future__ import annotations

from pipeline.kb import ingest_common


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
    """`RUN_STAT_KEYS` is what a user is told; `RUN_DIAG_KEYS` is what an engineer reads. One key
    in both would mean a counter can't be stripped from a user-facing payload without also
    stripping a diagnostic (or the reverse)."""
    assert not set(ingest_common.RUN_STAT_KEYS) & set(ingest_common.RUN_DIAG_KEYS)


def test_a_wrongly_shaped_diagnostic_is_dropped_not_forwarded(kb_home):
    """Each set has its OWN type filter, so a key carrying the wrong shape is dropped here rather
    than handed on to break whoever reads the report."""
    out = ingest_common.run_stats({"stage_latency": "0.4s", "added": {"n": 1}})
    assert out == {}
