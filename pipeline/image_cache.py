"""
pipeline/image_cache.py — the shared image-read cache (load / put / save).

One URL-keyed JSON file (`<state_dir>/image_descriptions.json`) recording what an image
read produced, so no image is ever paid for twice. Every reader in the system shares this
shape: the atom-KB ingesters (x / x-footprint / blog / substack / curation, plus
`pipeline/kb/vision.py`) and the legacy vault enrichment.

Why it lives at the top of `pipeline` and not inside `pipeline/processing/`: it used to sit
in `pipeline/processing/describe_images.py`, whose docstring argued for leaving it there —
"seven ingesters import it from this path and renaming the module would be churn with no
behavioral payoff." That was right at the time. It stopped being right when
`pipeline/processing/` became the vault producer package, queued for deletion: the churn now
buys the atom rail's independence from a package that is going away. Same reasoning that
moved rank fusion to `pipeline/rank.py`.

The lock exists because producers and the consumer share the dict. A producer adding a NEW
key while `save_image_cache` iterates it to JSON would raise "dict changed size during
iteration", so writes go through `cache_put` and the save takes a snapshot under the lock.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

_CACHE_LOCK = threading.RLock()


def load_image_cache(state_dir: Path) -> dict:
    cache_path = state_dir / "image_descriptions.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def cache_put(cache: dict, url: str, value) -> None:
    """Thread-safe store into an image-cache dict — the counterpart to save_image_cache's guarded
    snapshot, so a producer's write can't race the consumer's whole-dict JSON dump."""
    with _CACHE_LOCK:
        cache[url] = value


def save_image_cache(state_dir: Path, cache: dict) -> None:
    cache_path = state_dir / "image_descriptions.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:                    # snapshot under the lock, then write OUTSIDE it (I/O off-lock)
        snap = dict(cache)
    cache_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False))
