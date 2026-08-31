"""
pipeline/jitter.py — the one stable TTL jitter.

`stable_factor` turns any key into a multiplier inside a band, derived from the key and nothing
else. Every rail with a batch-stamped freshness clock needs it, for the same reason: a batch
stamped together falls due together, and the clustering does not decay — a batch refreshed together
gets re-stamped together, so the burst re-forms on every cycle. That is phase-locking, and at a
roster large enough that one tick's due set exceeds what a bounded run can drain, every cycle then
starts with a burst and a permanent backlog.

Derived, never drawn. A `random()` inside a staleness check makes staleness nondeterministic —
the exact property the repeat-run harnesses verify (N consecutive no-op runs, identical). Hashing
the key keeps the check a pure function of stored state, so an item's TTL is the same on every
call, in every process, forever.

Why it sits at the top of `pipeline` and not inside a rail: `oracle_refresh_state.jitter_factor`
was the first copy, and the candidate probe needs the identical primitive on a different key shape.
Copying it would have been the third fork of a shared primitive in this tree — after the timestamp
parse (three copies, one silently drifted) and `slugify` (seven, drifted on hyphen runs and
truncate order). The house rule those two wrote is that a shared primitive moves to a neutral
single-purpose module at the top of `pipeline/` and every rail imports from THERE:
`pipeline.rank`, `pipeline.sqlite_db`, `pipeline.timeparse`. This is the fourth.

(`pipeline.slug` was the fifth member of that list until 2026-08-30 and is now DELETED — see the
`retired-filename-slug` guard. Its merge was the right call and the house rule it wrote still
holds; what ended was the NEED, when the vault stopped being written and no caller had a filename
to slug any more. A neutral module earns its place from live importers, not from being neutral.)
"""

from __future__ import annotations

import hashlib


def stable_factor(key: str, band: float) -> float:
    """A multiplier in `[1-band, 1+band]`, derived from `key` alone.

    Pure and total: same key → same factor, in every process, forever. That is what lets it
    de-synchronize a batch without costing determinism. 16 bits of a sha256 is far more spread than
    any sane band can express, so the quantization is invisible."""
    h = hashlib.sha256(key.encode()).digest()
    unit = int.from_bytes(h[:2], "big") / 65535.0           # [0, 1]
    return 1.0 + band * (2.0 * unit - 1.0)                  # [1-band, 1+band]
