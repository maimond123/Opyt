"""
pipeline/kb/onboard_state.py
Where onboarding stands, recomputed from disk on every call.

No state file (decision 5): a prior phase-machine file could silently claim retired phases were
done. The `retired-onboarding-state-file` guard bans rebuilding it — this is a derived view,
never stored state.

The exceptions are the two MARKERS, and they are markers rather than phase files: each records
that a QUESTION WAS PUT, never what the answer was.

  • `onboard_consent_asked` — a user who answers "no to both" leaves zero consent markers,
    byte-identical to never having been asked, so without it `onboard` re-asks forever.
  • `onboard_sources_skipped` — same shape, for "I do not want to connect anything yet". Every
    other way out of the sources phase is a real disk fact; this one has none by definition.
"""
from __future__ import annotations

from pathlib import Path

from opyt_core import readiness
from opyt_core.paths import opyt_path

ASKED_MARKER = "onboard_consent_asked"
SOURCES_SKIPPED_MARKER = "onboard_sources_skipped"

# What `_store_sources` reports when the store cannot be read. Every value False, so a broken
# read sends the user to the sources phase, which re-checks — never to `done`, which would skip
# a real step.
_NO_STORE_SOURCES = {"oracles": False, "watchlist": False, "candidates": False}


def _connected_x() -> bool:
    """Whether OPYT's own persistent browser profile is logged into X."""
    from pipeline.ingestion.x_graphql import has_managed_x_session
    return has_managed_x_session()


def _connected_substack() -> bool:
    """Whether OPYT's own persistent browser profile is logged into Substack.

    Deliberately NOT the reader the Substack collector uses. That one may read the user's own
    browser; this one runs before consent exists and scans only profiles OPYT created. See
    `sources.substack.has_managed_substack_session` for the full split.
    """
    from pipeline.ingestion.sources.substack import has_managed_substack_session
    return has_managed_substack_session()


def _store_sources() -> dict:
    """The three roots that leave a row rather than a session: people the user named, topics the
    user put on the watchlist, and candidates a collector already surfaced.

    Existence, not counts. The phase asks "will anything produce atoms or candidates", and one
    row answers it as well as a thousand.
    """
    from pipeline.kb import frontier_queries, schema
    conn = schema.connect()
    try:
        def any_row(sql: str, *args) -> bool:
            return conn.execute(sql, args).fetchone() is not None
        return {
            "oracles": any_row("SELECT 1 FROM oracles LIMIT 1"),
            "watchlist": any_row("SELECT 1 FROM frontier_queries WHERE generator=? LIMIT 1",
                                 frontier_queries.USER_GENERATOR),
            "candidates": any_row("SELECT 1 FROM curation_signals LIMIT 1"),
        }
    finally:
        conn.close()


def _curation_ok() -> bool:
    from pipeline.kb import curation_state, schema
    conn = schema.connect()
    try:
        return any(r.ok for r in curation_state.list_runs(conn))
    finally:
        conn.close()


def _curation_pending(connected: set[str]) -> list[str]:
    """Connected platforms whose OWN collectors have never succeeded.

    ⚠️ PER PLATFORM, BECAUSE PLATFORMS ARE CONNECTED ONE AT A TIME. The phase gate was
    `collectors and not _curation_ok()` — "does ANY collector anywhere have a successful run" —
    so the first success on the first platform flipped the phase to `done` permanently and every
    platform connected afterwards was orphaned. Measured 2026-09-13: David connected Substack,
    `substack_follows` succeeded, then he connected X and `x_lists`/`x_following`/`x_likes` never
    ran once. `collector_runs` held only Substack rows while `x_lists` returned 6 candidates the
    moment it was invoked by hand — a valid session, silently never read.

    That is a SCORING fault, not a missing nicety: candidates are ranked on distinct
    (signal_type, platform) pairs, so an unread platform cannot corroborate anyone, and nothing
    pre-ticks.

    Keyed off `COLLECTOR_SPECS`, so a platform that gains or loses a collector needs no edit here.
    """
    from pipeline.kb import curation_state, ingest_curation, schema
    conn = schema.connect()
    try:
        ok = {r.collector for r in curation_state.list_runs(conn) if r.ok}
    finally:
        conn.close()
    read = {s.platform for s in ingest_curation.COLLECTOR_SPECS if s.collector in ok}
    return sorted(connected - read)


def _consent_markers() -> tuple[bool, bool, bool]:
    """(bookmark, substack_saved, oracle_refresh) — does each rail's OWN marker file exist?

    Ask the rail, never re-derive the path. Each rail resolves its marker through an env override
    ($OPYT_BOOKMARK_CATCHUP_CONSENT / $OPYT_SUBSTACK_SAVED_CATCHUP_CONSENT /
    $OPYT_ORACLE_REFRESH_CONSENT) falling back to `opyt_path(...)`. Spelling
    `opyt_path("bookmark_catchup_consent")` here would be a SECOND path to the same fact that
    silently disagrees the moment an override is set — the same shape as the five credential
    registries.

    This reports the MARKER, not `<rail>.consented()`, which for two of the three also returns
    True for any established store. Those are different questions: "did the user opt in" vs "may
    this rail run". `onboard` is asking the first one — which is why the Substack marker's
    absence on a store full of atoms is meaningful rather than moot.
    """
    from pipeline.kb import bookmark_catchup, oracle_refresh, substack_saved_catchup
    return (bookmark_catchup._consent_marker().exists(),
            substack_saved_catchup._consent_marker().exists(),
            oracle_refresh._consent_marker().exists())


def _marker(name: str) -> Path:
    return opyt_path(name)


def _put(name: str) -> None:
    """Record that a question was put. Best-effort: a home that cannot be written re-asks, which
    is the harmless direction."""
    try:
        p = _marker(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def _asked() -> bool:
    return _marker(ASKED_MARKER).exists()


def _skipped() -> bool:
    return _marker(SOURCES_SKIPPED_MARKER).exists()


def mark_asked() -> None:
    _put(ASKED_MARKER)


def mark_sources_skipped() -> None:
    _put(SOURCES_SKIPPED_MARKER)


def _safe(fn, default):
    """Fail-safe: a broken probe must not crash the tool. It reads as not done, which sends the
    user to a phase that will re-check — never as DONE, which would skip a real step."""
    try:
        return fn()
    except Exception:
        return default


def derive() -> dict:
    """The whole onboarding picture: the one required key, what this home can read from, whether
    the consent question was put, and whether a collector has ever produced candidates.

    The `sources` phase replaced a `browser` phase on 2026-09-07. That phase asked "is X
    connected" and had no bypass, so a user who reads Substack, or named blogs, or watches
    research topics sat there forever and never reached consent or curation. This one asks
    whether ANYTHING in this home will produce atoms or candidates, and X is one of the answers
    rather than the question.
    """
    orx = _safe(readiness.openrouter, {"state": "unknown", "message": "probe failed"})
    x = _safe(_connected_x, False)
    substack = _safe(_connected_substack, False)
    store = _safe(_store_sources, _NO_STORE_SOURCES)
    skipped = _safe(_skipped, False)
    asked = _safe(_asked, False)
    curated = _safe(_curation_ok, False)
    bm_consent, saved_consent, or_consent = _safe(_consent_markers, (False, False, False))

    # ONE key. `TWITTERAPI_KEY` was the second until 2026-08-30; X reads now run on the browser
    # session this same function probes below, so the X requirement moved from the keys phase to
    # the sources phase rather than disappearing.
    keys_ok = orx["state"] == "ok"

    # Is a platform with AUTO-DISCOVERY connected? Only those have collectors, so only they can
    # make the curation phase mean anything. The blog and topic roots have no account to read —
    # that is a property of those roots, not a gap — so for a user who took one of them the
    # curation phase has nothing to run and must not gate `done`.
    collectors = x or substack
    connected = collectors or any(store.values())
    # Which of the auto-discovery platforms are connected RIGHT NOW, and which of those have
    # never had their own collectors read. `pending` is what the curation phase is for.
    live = {p for p, on in (("x", x), ("substack", substack)) if on}
    pending = _safe(lambda: _curation_pending(live), [])

    out = {
        "keys": {"openrouter": orx, "ok": keys_ok},
        "sources": {"x": x, "substack": substack, **store,
                    "skipped": skipped, "ok": connected},
        "consent": {"asked": asked, "bookmark": bm_consent,
                    "substack_saved": saved_consent, "oracle_refresh": or_consent},
        "curation": {"any_ok": curated, "applicable": collectors, "pending": pending},
    }
    if not keys_ok:
        out["phase"] = "keys"
    elif not (connected or skipped):
        out["phase"] = "sources"
    elif not asked:
        out["phase"] = "consent"
    elif pending:
        out["phase"] = "curation"
    else:
        out["phase"] = "done"
    return out
