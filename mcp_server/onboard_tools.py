"""
mcp_server/onboard_tools.py — `onboard`: ready the machine, then hand off to `oracle`.

A thin orchestrator: it sequences existing screen/confirm/ingest code rather than
re-implementing it.

The channel rule: secrets travel over loopback (opyt_core/local_auth.py); decisions travel
over chat as arguments. No parameter of this tool may ever carry a credential — it would land
in the transcript and every later turn's re-send.

Semantic Scholar is not mentioned to the user: AI2 no longer approves third-party key requests,
so its row in `opyt_core/credentials_registry.py` stays present but unadvertised, referred to by
SERVICE name only (this module does not load env).
"""
from __future__ import annotations

from pathlib import Path

from opyt_core import key_paste, openrouter_oauth
from pipeline.kb import onboard_state


def register_onboard_tools(mcp) -> None:

    @mcp.tool()
    def onboard(browser_profile: str | None = None, consent: str | None = None,
                skip_github: bool = True, guided_login: bool = False) -> dict:
        """**Set up OPYT on this machine.** Call this first on a fresh install, and any time
        setup looks incomplete. Idempotent and re-entrant — it recomputes where you are from
        disk on every call, so calling it twice never repeats a finished step.

        It runs in up to three calls, because two steps wait on a human:
          1. **OpenRouter** — a browser tab opens; click Approve. Nothing to paste.
          2. **Your browser session** — reads your own logged-in X cookies, on this machine
             only. This is the ONLY way OPYT reads X: there is no API key and no third party.
             If no browser holds an X session, Opyt can open one for you to log into
             (`guided_login=true`). A couple of browsers still trip a native consent prompt
             (Full Disk Access for Safari, Keychain for Arc); when one applies, this tool warns
             you before it appears.
          3. **Consent** — one question, two commitments, and you may answer them separately.

        Arguments are decisions, never credentials:
          • `browser_profile` — which Chrome profile holds your X session, when several are
            logged in. The answer is remembered in settings.yaml.
          • `guided_login` — accept the offer to have Opyt open a browser window you log
            into, when no browser on this machine has an X session.
          • `consent` — one of `both` | `backlog` | `refresh` | `none`.
          • `skip_github` — GitHub is optional and skipped by default.

        Then call `oracle` to choose who to trust.
        """
        # Skip the browser probe here — it is the first look at the user's browser, and on a
        # machine with a gated backend (Safari, Arc) it can raise a native dialog before the
        # user has been warned it is coming.
        state = onboard_state.derive(probe_browser=False)

        if state["phase"] == "keys":
            return _phase_keys(state, skip_github=skip_github)

        # Re-entry exception: with both keys already present, this is the first call and there
        # is no prior payload to carry the warning on, so the whole body is one line that warns
        # and reads nothing.
        if not _warned():
            _mark_warned()
            return _prewarn_only()

        state = onboard_state.derive(probe_browser=True)   # now the probe is expected

        # `browser_profile=` is honored even when the browser step is already settled, since
        # naming a profile IS the decision — gating on phase alone would block changing it later.
        if state["phase"] == "browser" or browser_profile or guided_login:
            asked = _phase_browser(state, browser_profile=browser_profile,
                                   guided_login=guided_login)
            if asked is not None:
                return asked
            # Settling the browser step IS the consent to read this user's logged-in session, so
            # it is where the curation rail's gate opens. Granted here rather than in
            # `_apply_consent` on purpose: that question is about spending money, and this one is
            # not — bundling them would be the silent cross-opt-in `bookmark_catchup.consented`
            # warns about.
            from pipeline.kb import curation_catchup
            curation_catchup.grant_consent()
            state = onboard_state.derive(probe_browser=True)

        if state["phase"] == "consent" or consent:
            if consent is None:
                return _consent_prompt()
            if consent not in _CONSENT_WORDS:
                # Refuse, never guess. "yes" is ambiguous across two commitments with
                # different cost shapes, and one of them cannot be revoked.
                return {"status": "error", "phase": "consent",
                        "message": (f"`consent={consent!r}` is not one of "
                                    f"{' | '.join(_CONSENT_WORDS)}. Nothing was recorded.")}
            applied = _apply_consent(consent)
            state = onboard_state.derive(probe_browser=True)
            state["consent_applied"] = applied

        extra = ({"consent_applied": state["consent_applied"]}
                 if "consent_applied" in state else {})

        if state["phase"] == "curation":
            ran = _run_curation()
            state = onboard_state.derive(probe_browser=True)
            extra["curation"] = ran

        return _handoff(state, **extra)


def _phase_keys(state: dict, *, skip_github: bool) -> dict:
    orx = state["keys"]["openrouter"]

    # OpenRouter is the ONLY key. An UNFUNDED account blocks exactly like a missing one: pulling
    # into a store that cannot be indexed is the fail-safe violation.
    if orx["state"] in ("unfunded", "dead", "unknown"):
        return {"status": "blocked", "phase": "keys", "openrouter": orx["state"],
                "message": (f"{orx['message']} Nothing was pulled and nothing was spent. "
                            f"⚠️ The four curation collectors this would have run are FREE — "
                            f"they cost no API credits at all. They are still blocked, because "
                            f"anything they collect cannot be embedded or searched without a "
                            f"working OpenRouter key, and a store you cannot query is not worth "
                            f"building.")}
    if orx["state"] == "missing":
        got = openrouter_oauth.acquire()
        return {"status": got["status"], "phase": "keys", "step": "openrouter",
                "message": got["message"], "next": _consent_prewarn_step()}

    return {"status": "ok", "phase": "keys", "message": "The one required key is live.",
            "next": _consent_prewarn_step()}


# The filename keeps its historical "keychain" spelling on purpose: existing installs already
# carry the marker file, and renaming it would re-warn every one of them once.
WARNED_MARKER = "onboard_keychain_warned"


def _warned() -> bool:
    """Has the consent pre-warn already been shown? A marker of what the user has been TOLD,
    not a phase file — without it the re-entry path would warn forever and never read."""
    from opyt_core.paths import opyt_path
    try:
        return opyt_path(WARNED_MARKER).exists()
    except Exception:
        return False


def _mark_warned() -> None:
    from opyt_core.paths import opyt_path
    try:
        p = opyt_path(WARNED_MARKER)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def _save_profile(name: str) -> None:
    """Persist the profile pick to `cookies.profile` in the user's settings.yaml.

    Resolves a **write** path, not `config_path()` — the read resolver falls back to the
    repo/packaged settings.yaml, which is read-only in a pip install. Best-effort: a config
    write must never break setup, since the env var still overrides
    """
    import os

    import yaml

    from opyt_core import bootstrap, config
    try:
        bootstrap.ensure_config()
        target = (Path(os.environ["OPYT_CONFIG"]).expanduser()
                  if os.environ.get("OPYT_CONFIG")
                  else config.opyt_home() / "settings.yaml")
        cur = yaml.safe_load(target.read_text()) if target.exists() else {}
        cur = cur if isinstance(cur, dict) else {}
        cur.setdefault("cookies", {})
        if not isinstance(cur["cookies"], dict):
            cur["cookies"] = {}
        cur["cookies"]["profile"] = name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(cur, sort_keys=False), encoding="utf-8")
    except Exception:
        pass


def _prewarn_only() -> dict:
    """A phase whose entire body is one sentence. It warns and reads NOTHING.

    The native-dialog half of the warning is conditional (see `_consent_prewarn_step`), but the
    round trip is not: it is where the user learns an agent is about to read their browser at
    all, which is the consent that matters whether or not macOS also asks."""
    return {"status": "warned", "phase": "browser", "reads_nothing_yet": True,
            "message": (_consent_prewarn_step().get("say_this_before_the_next_call")
                        or ("Next I read your own logged-in X session out of your browser — "
                            "nothing is read until you call again, and nothing leaves your "
                            "machine.")) + " Call `onboard` again when you are ready."}


# Where guided login sends the user. X is the only site `onboard` sets up, so the URL is the
# caller's to know — `guided_login` stays site-agnostic.
_X_LOGIN_URL = "https://x.com/login"


def _guided_login_step(accepted: bool) -> dict:
    """The no-session branch: three states, because "no X session" is three situations.

    Opyt can open a browser on a profile it owns and let the user log in THERE, so a user with
    no local X session finishes setup instead of being sent away to do it elsewhere. The window
    outlives this call (a tool call cannot block on a human), and the session is picked up on
    the next one — `browser_cookies` enumerates that profile like any other."""
    from pipeline.ingestion import browser_cookies as bc, guided_login
    from pipeline.ingestion.utils import SyncAuthError

    if accepted:
        try:
            backend = guided_login.start(_X_LOGIN_URL)
        except SyncAuthError as e:
            return {"status": "needs_login", "phase": "browser", "message": str(e)}
        return {"status": "awaiting_login", "phase": "browser",
                "message": (f"{backend.label} is open at x.com/login. Log in there — it is a "
                            f"separate profile Opyt owns, so it does not touch your normal "
                            f"{backend.label} windows, tabs or logins. When you are logged in, "
                            f"call `onboard` again.")}

    if bc.opyt_session_backends():
        return {"status": "awaiting_login", "phase": "browser",
                "message": ("The browser window Opyt opened has no X session in it yet. Finish "
                            "logging in there, then call `onboard` again. If you closed the "
                            "window, call `onboard` with `guided_login=true` to reopen it.")}

    return {"status": "needs_login", "phase": "browser",
            "message": ("No logged-in X session found in your browser. Two ways forward: log "
                        "into x.com in Chrome (or your usual browser) and call `onboard` "
                        "again, or call `onboard` with `guided_login=true` and Opyt will open "
                        "a browser window for you to log in — in a profile it owns, separate "
                        "from your normal windows and logins.")}


def _phase_browser(state: dict, *, browser_profile: str | None,
                   guided_login: bool = False) -> dict | None:
    """Settle which browser profile holds the user's X session. The pre-warn has already been
    shown by the time this runs — the caller gates on `_warned()`.

    Returns a payload when it must ASK or REPORT, and **None** when the step is settled, so the
    caller falls through to the next phase in the same call instead of costing a round trip.
    """
    profiles = state["browser"]["profiles"]
    blocked = state["browser"]["blocked"]

    if not profiles and blocked:
        # Reuse `remediation`'s copy verbatim rather than forking it here.
        from pipeline.ingestion import browser_cookies as bc
        return {"status": "blocked", "phase": "browser",
                "message": bc.remediation(blocked.get("kind", "other"),
                                          bc.backend_for(blocked.get("browser", "")), "X")}

    if not profiles:
        return _guided_login_step(guided_login)

    if browser_profile:
        chosen = next((p for p in profiles if p["profile"] == browser_profile), None)
        if chosen:
            _save_profile(chosen["profile"])
            return None
        # Refuse, don't persist — a saved profile with no X session breaks every later cookie
        # read, surfacing far from here as "not logged in".

    elif len(profiles) == 1:
        _save_profile(profiles[0]["profile"])      # remember the auto-pick; it is still a pick
        return None

    return {"status": "needs_choice", "phase": "browser",
            "choices": [{"profile": p["profile"], "label": p["label"]} for p in profiles],
            "message": ("Several browser profiles are logged into X. Call `onboard` again with "
                        "`browser_profile=` set to the one to use — the answer is remembered in "
                        "settings.yaml, so this is asked once."
                        + (f" `{browser_profile}` is not one of them."
                           if browser_profile else ""))}


def _consent_prewarn_step() -> dict:
    """The pre-warn, which must ride on the previous call's return. Reading a browser's cookie
    store can trigger a native consent dialog (Safari → Full Disk Access, Arc → Keychain) from a
    background subprocess, indistinguishable from malware unless the user is warned first.

    The copy is `prewarn_installed()`'s — derived from the backends actually on this machine, so
    it names the right browser and the right dialog (the hardcoded Chrome/Keychain copy this
    replaced was wrong on four of five installed-browser cases). {} when no installed backend
    trips a native gate, which is now the COMMON case: Chrome/Brave/Edge/Vivaldi/Opera are read
    by launching the browser itself, so they raise no dialog at all. Do not describe a dialog
    that never comes."""
    from pipeline.ingestion.browser_cookies import prewarn_installed
    msg = prewarn_installed()
    if not msg:
        return {}
    return {"step": "browser_session",
            "say_this_before_the_next_call":
                f"Next I read your own logged-in X session out of your browser. {msg}"}


# ── phase 2: consent ────────────────────────────────────────────────────────────────────────
#
# ONE question, TWO commitments, a split answer allowed, and BOTH cost shapes stated — or it is
# not consent to the recurring half.

_CONSENT_WORDS = {"both": (True, True), "backlog": (True, False),
                  "refresh": (False, True), "none": (False, False)}

# Quote the bracket, not the rate — a rate is unusable without an item count.
_MONTHLY_AT_50 = "$0.09"


def _consent_prompt() -> dict:
    return {
        "status": "needs_consent", "phase": "consent",
        "answers": list(_CONSENT_WORDS),
        "message": (
            "One question, two separate commitments — answer them together or separately by "
            "calling `onboard` again with `consent=both | backlog | refresh | none`.\n\n"
            "1. **Import your X bookmark backlog now** — ONE-TIME and bounded. It runs as soon "
            "as you say yes, against a $1.00/day ceiling.\n"
            "2. **Keep your Oracles current** — RECURRING, forever. About $0 today because your "
            f"roster is empty, rising to roughly {_MONTHLY_AT_50}/month at 50 Oracles.\n\n"
            "⚠️ These are not symmetrical, and you should know which is which before you "
            "answer. The recurring half can be switched off later (`consent=backlog` or "
            "`consent=none` revokes it). The one-time import CANNOT be turned off once it has "
            "run — there is no revoke for it, and once you have content in the store it counts "
            "as consented from then on. It is one-time and bounded, so the blast radius is "
            "small, but the choice is not reversible the way the other one is."),
    }


def _spawn_bookmarks() -> bool:
    from pipeline.kb.bookmark_catchup import spawn_bookmark_catchup
    return spawn_bookmark_catchup(force=True)


def _spawn_refresh() -> bool:
    from pipeline.kb.oracle_refresh import spawn_oracle_refresh
    return spawn_oracle_refresh(force=True)


def _confirmed_oracles() -> int:
    """How many Oracles are confirmed right now. Fail-safe: unreadable store reads as zero,
    which SUPPRESSES a spawn — the safe direction, since a spawn costs money."""
    try:
        from pipeline.kb import oracles, schema
        conn = schema.connect()
        try:
            return len(oracles.confirmed_oracles(conn))
        finally:
            conn.close()
    except Exception:
        return 0


def _apply_consent(word: str) -> dict:
    """Write the two markers, then spawn a rail IFF that rail has work right now — the session-
    open spawners alone would leave "import now" meaning nothing until the next coalesce window.
    """
    from pipeline.kb import bookmark_catchup, oracle_refresh
    want_backlog, want_refresh = _CONSENT_WORDS[word]
    spawned = []

    if want_backlog:
        bookmark_catchup.grant_consent()
        if _spawn_bookmarks():
            spawned.append("bookmark_catchup")

    if want_refresh:
        oracle_refresh.grant_consent()
        if _confirmed_oracles() > 0 and _spawn_refresh():
            spawned.append("oracle_refresh")
    else:
        # A toggle should toggle both ways — this is the only way to opt back out from chat.
        oracle_refresh.revoke_consent()

    onboard_state.mark_asked()
    return {"granted": [k for k, v in (("backlog", want_backlog),
                                       ("refresh", want_refresh)) if v],
            "started_now": spawned}


# ── phase 3: the free curation pull ─────────────────────────────────────────────────────────

def _run_curation() -> dict:
    """The four FREE people-only collectors: X Lists, following, likes, Substack subscriptions.

    This is `curation_catchup`, not `curation_pull` — the content-bearing arms of
    `curation_pull` cost money and time, so `onboard` opens the bookmark backlog's existing
    budgeted rail rather than carrying the pull itself.

    NEVER `curation_pull(tiered=True)` here: the ladder's gate reads the whole store's
    signalled-entity count, so on any established store it clears `sufficient_at` after Tier 1
    and permanently skips following and likes — the exact two collectors this needs.

    `force=True` bypasses the 6 h per-collector floor (meant for a background loop, not a fresh
    setup call) but not single-flight. It leaves a known Substack-saved-posts gap.
    """
    from pipeline.kb.curation_catchup import run_curation_catchup
    return run_curation_catchup(force=True)


# ── phase 4: derived progress, and the handoff ──────────────────────────────────────────────

def _candidate_count() -> int:
    """How many people the free pull surfaced for screening. Fail-safe: unreadable reads as 0,
    which understates progress rather than inventing it."""
    try:
        from pipeline.kb import schema, screen
        conn = schema.connect()
        try:
            return len(screen.rank_candidates(conn))
        finally:
            conn.close()
    except Exception:
        return 0


def _handoff(state: dict, **extra) -> dict:
    """What landed, and what to call next. No new state — everything here comes from `derive()`
    plus a candidate count. The copy bounds the first ingest deliberately, since
    `oracle(action='ingest')` is a synchronous foreground loop with no pick cap or time budget.
    """
    n = _candidate_count()
    done = state["phase"] == "done"
    return {
        "status": "ok" if done else "in_progress",
        "phase": state["phase"],
        "candidates": n,
        "next_tool": "oracle",
        "message": (
            f"Setup is complete. The free collectors surfaced {n} people to screen.\n\n"
            f"Next: call `oracle` to pick who to trust. Confirm THREE TO FIVE people first and "
            f"ingest those before adding more — the first ingest runs in the foreground and "
            f"walks every person you confirm, so a large first batch means a long wait with no "
            f"partial result."
            if done else
            f"Setup is not finished yet — currently at the `{state['phase']}` step. "
            f"Call `onboard` again to continue."),
        **extra,
    }
