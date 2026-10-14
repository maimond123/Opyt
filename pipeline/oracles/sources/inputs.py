"""
pipeline/oracles/sources/inputs.py — what a document worker loads once from its context: its image
cache, which each worker owns (each web pool thread has its own context).
"""

from __future__ import annotations

from opyt_core.paths import opyt_home
from pipeline.image_cache import load_image_cache
from pipeline.image_cache import save_image_cache
from pipeline.worker.quota_loop.source import Context


def image_cache(ctx: Context) -> dict:
    """This worker's image cache, topped up from what is on disk."""
    cache = ctx.cache.setdefault("images", {})
    cache.update(load_image_cache(opyt_home()))
    return cache


def save_images(cache):
    home = opyt_home()
    save_image_cache(home, cache)
