"""
pipeline/kb/onboard_state.py
Where onboarding stands, recomputed from disk on every call.

No state file (decision 5): a prior phase-machine file could silently claim retired phases were
done. The `retired-onboarding-state-file` guard bans rebuilding it — this is a derived view,
never stored state.

The one exception is `onboard_consent_asked`, and it is a MARKER, not a phase file. A user who
answers "no to both" leaves zero consent markers — byte-identical to never having been asked — so
without it `onboard` re-asks a settled question forever. It records that the question was PUT,
never what the answer was.
"""
from __future__ import annotations

from opyt_core import readiness
from opyt_core.paths import opyt_path

ASKED_MARKER = "onboard_consent_asked"


def _x_profiles():
    """(profiles, blocked_reason). Fail-safe: any error reads as "none, unknown reason".

    This is the first look at the user's browser, and on a machine with a gated backend
    (Safari's Full Disk Access, Arc's Keychain) it can raise a native dialog. Never call it
    before the pre-warn has been shown — see `derive(probe_browser=False)`.
    """
    from pipeline.ingestion.x_graphql import list_x_logged_in_profiles
    return list_x_logged_in_profiles(), None


def _curation_ok() -> bool:
    from pipeline.kb import curation_state, schema
    conn = schema.connect()
    try:
        return any(r.ok for r in curation_state.list_runs(conn))
    finally:
        conn.close()


def _consent_markers() -> tuple[bool, bool]:
    """(bookmark, oracle_refresh) — does each rail's OWN marker file exist?

    Ask the rail, never re-derive the path. Both rails resolve their marker through an env
    override ($OPYT_BOOKMARK_CATCHUP_CONSENT / $OPYT_ORACLE_REFRESH_CONSENT) falling back to
    `opyt_path(...)`. Spelling `opyt_path("bookmark_catchup_consent")` here would be a SECOND
    path to the same fact that silently disagrees the moment an override is set — the same
    shape as the five credential registries.

    This reports the MARKER, not `bookmark_catchup.consented()`, which also returns True for any
    established store. Those are different questions: "did the user opt in" vs "may this rail
    run". `onboard` is asking the first one.
    """
    from pipeline.kb import bookmark_catchup, oracle_refresh
    return (bookmark_catchup._consent_marker().exists(),
            oracle_refresh._consent_marker().exists())


def _chosen_profile() -> str | None:
    """The profile already pinned in settings.yaml/$X_CHROME_PROFILE, if any. Asks the one
    resolver rather than re-reading the file — see `opyt_core.config.cookie_profile`."""
    from opyt_core.config import cookie_profile
    return cookie_profile()


def _asked() -> bool:
    return opyt_path(ASKED_MARKER).exists()


def mark_asked() -> None:
    try:
        p = opyt_path(ASKED_MARKER)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def _safe(fn, default):
    """Fail-safe: a broken probe must not crash the tool. It reads as not done, which sends the
    user to a phase that will re-check — never as DONE, which would skip a real step."""
    try:
        return fn()
    except Exception:
        return default


def derive(*, probe_browser: bool = True) -> dict:
    """The whole onboarding picture. `probe_browser=False` skips the browser scan (and so any
    native dialog it could raise) — the tool uses it to render the pre-warn first."""
    orx = _safe(readiness.openrouter, {"state": "unknown", "message": "probe failed"})
    profiles, blocked = _safe(_x_profiles, ([], None)) if probe_browser else ([], None)
    asked = _safe(_asked, False)
    curated = _safe(_curation_ok, False)
    bm_consent, or_consent = _safe(_consent_markers, (False, False))

    # ONE key. `TWITTERAPI_KEY` was the second until 2026-08-30; X reads now run on the browser
    # session this same function probes below, so the X requirement moved from the keys phase to
    # the browser phase rather than disappearing.
    keys_ok = orx["state"] == "ok"

    # "A profile is logged in" is not the same as "the browser step is done". With several
    # logged-in profiles nothing is settled until the user picks one, and a pick that no longer
    # matches any logged-in profile is not settled either. Treating any non-empty list as done
    # would silently skip the question and leave `pick()` to raise on ambiguity later, far from
    # the call that could have asked.
    chosen = _safe(_chosen_profile, None)
    browser_ok = bool(profiles) and (
        len(profiles) == 1 or any(p.get("profile") == chosen for p in profiles))

    out = {
        "keys": {"openrouter": orx, "ok": keys_ok},
        "browser": {"profiles": [{"profile": p["profile"], "label": p["label"]}
                                 for p in profiles],
                    "blocked": blocked, "probed": probe_browser,
                    "chosen": chosen, "ok": browser_ok},
        "consent": {"asked": asked, "bookmark": bm_consent, "oracle_refresh": or_consent},
        "curation": {"any_ok": curated},
    }
    if not keys_ok:
        out["phase"] = "keys"
    elif probe_browser and not browser_ok:
        out["phase"] = "browser"
    elif not asked:
        out["phase"] = "consent"
    elif not curated:
        out["phase"] = "curation"
    else:
        out["phase"] = "done"
    return out
