"""Resolve and persist operation identifiers independently of X's transport."""

import json
import os
from pathlib import Path
from typing import Callable

from opyt_core.paths import opyt_home
from pipeline.ingestion.utils import log


def query_id_choice(override: str | None, seed: str | None,
                    cached: str | None) -> str | None:
    """Choose the first configured identifier, or return None when discovery is needed."""
    return next((value for value in (override, seed, cached) if value), None)


def _qid_cache_path() -> Path:
    return opyt_home() / "x_query_ids.json"


def qid_cache_get(op: str) -> str | None:
    try:
        return json.loads(_qid_cache_path().read_text()).get(op)
    except Exception:
        return None


def qid_cache_put(op: str, qid: str) -> None:
    path = _qid_cache_path()
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data[op] = qid
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception as error:
        log(f"[x-graphql] could not cache queryId: {error}")


def qid_cache_clear(op: str) -> None:
    path = _qid_cache_path()
    try:
        data = json.loads(path.read_text())
        data.pop(op, None)
        path.write_text(json.dumps(data))
    except Exception:
        pass


def cached_query_id(op: str, discover: Callable[[], str | None]) -> str | None:
    """Read this operation's cache, discovering and storing one id only on a miss."""
    cached = qid_cache_get(op)
    if cached:
        return cached
    discovered = discover()
    if discovered:
        qid_cache_put(op, discovered)
    return discovered


def resolve(op: str, *, default_seed: str = "", env_var: str | None = None,
            discover: Callable[[], str | None]) -> str | None:
    """Apply override/seed precedence, then use the cached discovery job; None when nothing
    resolves."""
    override = os.getenv(env_var) if env_var else None
    selected = query_id_choice(override, default_seed, None)
    if selected:
        return selected
    return cached_query_id(op, discover)
