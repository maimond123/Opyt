"""
pipeline/image_cache.py — the shared image-read cache (load / put / save).

One URL-keyed JSON file (`<state_dir>/image_descriptions.json`) recording what an image read
produced, so no image is paid for twice. Producers write through `cache_put` and the save takes a
snapshot under the same lock, so a new key never lands while the dict is being dumped.
"""

from __future__ import annotations

import json
import threading
import fcntl
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

_CACHE_LOCK = threading.RLock()


def load_image_cache(state_dir: Path) -> dict:
    cache_path = state_dir / "image_descriptions.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def cache_put(cache: dict, url: str, value) -> None:
    """Thread-safe store into an image-cache dict; pairs with save_image_cache's guarded snapshot."""
    with _CACHE_LOCK:
        cache[url] = value


def save_image_cache(state_dir: Path, cache: dict) -> None:
    cache_path = state_dir / "image_descriptions.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Different quota workers, and the bookmark import process, can finish together.
    # Serialize read/merge/replace; readers see a complete JSON file on either side of replace.
    with _CACHE_LOCK:
        snap = dict(cache)
        with (state_dir / "image_descriptions.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = {**load_image_cache(state_dir), **snap}
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", dir=state_dir, prefix=".images-", delete=False) as out:
                    temporary = out.name
                    json.dump(merged, out, indent=2, ensure_ascii=False)
                os.replace(temporary, cache_path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
                fcntl.flock(lock, fcntl.LOCK_UN)
