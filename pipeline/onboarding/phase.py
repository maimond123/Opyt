"""
pipeline/onboarding/phase.py — where setup stands, and the step one `onboard` call may take from
there.

The phases are sources → curation → done, derived from disk on every call and never
stored. The one exception is the `onboard_sources_skipped` marker, which records that the sources
question was put and answered "not yet"; `skip_sources` is its only writer.
"""
from __future__ import annotations

from enum import Enum

from opyt_core.paths import opyt_path
from pipeline.curation import collectors
from pipeline.kb import schema
from pipeline.onboarding import roots
from pipeline.platforms import connections
from pipeline.topics import store as topics_store


def decide(*, connected: bool, skipped: bool, pending: list[str]) -> str:
    """The phase, from what the probes found. The first phase still outstanding wins."""
    if not (connected or skipped):
        return "sources"
    if pending:
        return "curation"
    return "done"


class Step(Enum):
    """What one call does when it carries no `start` word."""
    REFUSE_SOURCE = "refuse_source"
    SKIP = "skip"
    CONNECT = "connect"
    NAME_ROOT = "name_root"
    SOURCES_MENU = "sources_menu"
    CURATION = "curation"
    HANDOFF = "handoff"


def next_step(phase: str, source: str | None) -> Step:
    """The step for this call. An explicit `source` is honored at any phase, not only while the
    sources step is outstanding: a connected session still expires, and can belong to the wrong
    account."""
    if source is not None:
        if source not in roots.SOURCE_WORDS:
            return Step.REFUSE_SOURCE
        if source == "skip":
            return Step.SKIP
        return Step.CONNECT if source in roots.LOGIN_URLS else Step.NAME_ROOT
    if phase == "sources":
        return Step.SOURCES_MENU
    if phase == "curation":
        return Step.CURATION
    return Step.HANDOFF


SOURCES_SKIPPED_MARKER = "onboard_sources_skipped"


# What `_store_sources` reports when the store cannot be read. Every value False, so a broken
# read sends the user to the sources phase, which re-checks — never to `done`, which would skip
# a real step.
_NO_STORE_SOURCES = {"oracles": False, "topics": False, "candidates": False}


def derive() -> dict:
    """The whole onboarding picture: what this home can read from, and whether each connected
    platform's collectors have produced candidates yet."""
    # Scans only profiles OPYT created; see `connections._connected_substack` for why it is not
    # the collector's reader.
    logins = {"x": _safe(connections._connected_x, False),
              "substack": _safe(connections._connected_substack, False)}
    store = _safe(_store_sources, _NO_STORE_SOURCES)
    skipped = _safe(_skipped, False)

    live = roots.connected(logins)
    # Only auto-discovery platforms have collectors, so only they can gate `done` on curation.
    applicable = bool(live)
    connected = applicable or any(store.values())
    # Live platforms whose collectors have never read.
    pending = _safe(lambda: collectors.curation_pending(set(live)), [])

    out = {
        "sources": {**logins, **store,
                    "skipped": skipped, "ok": connected},
        "curation": {"applicable": applicable, "pending": pending},
    }
    out["phase"] = decide(connected=connected, skipped=skipped, pending=pending)
    return out


def _safe(fn, default):
    """Fail-safe: a broken probe reads as not done, sending the user to a phase that re-checks
    rather than skipping a real step."""
    try:
        return fn()
    except Exception:
        return default


def _store_sources() -> dict:
    """The three roots that leave a row rather than a session: people the user named, a topic they
    follow, and candidates a collector already surfaced. Existence, not counts. A topic no longer
    followed is not a source."""
    conn = schema.connect()
    try:
        def any_row(sql: str, *args) -> bool:
            return conn.execute(sql, args).fetchone() is not None
        return {
            "oracles": any_row("SELECT 1 FROM oracles LIMIT 1"),
            "topics": bool(topics_store.topics(conn)),
            "candidates": any_row("SELECT 1 FROM curation_signals LIMIT 1"),
        }
    finally:
        conn.close()


def _skipped() -> bool:
    return opyt_path(SOURCES_SKIPPED_MARKER).exists()


def _put(name: str) -> None:
    """Record that a question was put. Best-effort: a home that cannot be written re-asks, which
    is the harmless direction."""
    try:
        p = opyt_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def skip_sources() -> tuple[dict, Step]:
    """Record that the sources question was put and answered "not yet", then re-derive. Returns
    the new state and the step it calls for."""
    _put(SOURCES_SKIPPED_MARKER)
    state = derive()
    return state, next_step(state["phase"], None)
