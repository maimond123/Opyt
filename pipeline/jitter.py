"""
pipeline/jitter.py — the one stable TTL jitter.

`stable_factor` turns a key into a multiplier inside a band, derived from the key alone. A batch
stamped together falls due together and gets re-stamped together, so without jitter the burst
re-forms every cycle. Hashing rather than `random()` keeps a staleness check deterministic.
"""

from __future__ import annotations

import hashlib


def stable_factor(key: str, band: float) -> float:
    """A multiplier in `[1-band, 1+band]`, derived from `key` alone: same key, same factor."""
    h = hashlib.sha256(key.encode()).digest()
    unit = int.from_bytes(h[:2], "big") / 65535.0           # [0, 1]
    return 1.0 + band * (2.0 * unit - 1.0)                  # [1-band, 1+band]
