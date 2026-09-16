"""
opyt_core/readiness.py
Is this install actually able to work? Presence is NOT readiness.

`keys.status()` answers "is a key SET". That question
green-lights an OpenRouter key with no credit behind it — which authenticates fine and then fails
every embed, so the store can be built and never queried. This module asks the real question by
spending a fraction of a cent on a one-token ping.

The OAuth flow makes the unfunded case more likely, not less. A pasted key came from a human
who opened the dashboard and might have noticed their balance. An OAuth key is minted
automatically at the end of a signup the user just completed. So the user most likely to hold an
unfunded key is exactly the one zero-paste serves best.

FIVE STATES SINCE 2026-09-11, because a key can now arrive two ways. `opyt_core/trial.py` mints
one against the gateway's own balance so a first run needs no account; when that one stops
working the user is not unfunded, they are OUT OF TRIAL, and the two want opposite sentences.
This module is where that fork is resolved, because it is the module that already owns "what is
wrong with the key and what does the person do about it".

A NUMBER THIS MODULE ASSERTS AND HAS NOT MEASURED: `COST_NOTE`. See the comment on it.
"""
from __future__ import annotations

TOPUP_URL = "https://openrouter.ai/credits"

# The ONE sentence that tells a user what this costs, rendered wherever they are sent to a
# payment page and nowhere else. It is a CLAIM about real spend, and it is stated here once so
# that correcting it is a one-line edit rather than a hunt through three prompts.
#
# MEASURED 2026-09-16, and the previous wording did not survive it. This said "a few pennies"
# from 2026-09-11, written to the product's intent with a ⚠️ on it saying the claim was unmeasured
# and would be wrong at the worst possible moment if real spend came back higher. It did: one
# hosted onboarding spent $0.2001 — see the re-measurement note on `gateway.trial.DEFAULT_LIMIT_USD`
# for what changed and why. Twenty pennies is not "a few pennies", and this string is rendered
# into `_trial_over_prompt` at the exact moment a user is deciding whether to trust OPYT with a
# card, so it said the wrong thing at that moment.
#
# NO FIGURE IS NAMED, deliberately, and `tests/kb/test_allowance_notice.py` enforces it. A literal
# "20 cents" would be stale at the next measurement and at every ceiling change, and it would be
# stale in a sentence about money. "Well under a dollar" is the weakest claim the measurement
# actually supports, which is what makes it the one that survives being re-measured.
#
# The claim covers BUILDING A FIRST CORPUS only. Nothing here says what staying current costs:
# the rails now run hourly, `candidate_probe` alone paces up to 120 candidates a day, and that
# has never been measured over a full day. Do not extend this sentence to ongoing use until it is.
COST_NOTE = ("For context on the amount: building a first corpus has measured well under a "
             "dollar of model usage — it is a small one-off top-up, not a subscription, and "
             "whatever is left stays on their OpenRouter balance.")


def _credential(service: str) -> str | None:
    from pipeline.credentials import get_credential
    return get_credential(service)


def _ping(key: str) -> tuple[bool, str, int | None]:
    from pipeline.llm_providers import validate_provider_status
    return validate_provider_status("openrouter", key)


def _is_trial() -> bool:
    """Whether OPYT minted the key this install holds. Indirected like `_credential` and `_ping`
    so a test can set it without a marker file on disk."""
    from opyt_core import trial
    return trial.is_trial()


def openrouter() -> dict:
    """{'state': missing|dead|unfunded|unknown|ok, 'message': str}. Never returns a key value."""
    if not _credential("openrouter"):
        # A probe result, not an instruction. What the user is TOLD about this step lives in
        # `onboard_tools._openrouter_prompt`, which is the surface that shows it to a person and
        # the only place that knows the step now takes two calls. Nothing renders this string —
        # `_phase_keys` returns `orx["message"]` only for the three states it BLOCKS on.
        return {"state": "missing", "message": "No OpenRouter key yet."}
    ok, why, status = _ping(_credential("openrouter"))
    if ok:
        return {"state": "ok", "message": "OpenRouter key is live and has credit."}

    # A TRIAL key that stopped working is a different event from a user key that stopped
    # working, and the difference is the whole reason `trial` writes a marker. OpenRouter
    # returns the SAME 402 for "this account has no credit" and "this key's cap is spent", and
    # 401/403 for a key that has expired or been revoked. For a user those mean "go fund your
    # account" and "re-authorize"; for a trial they both mean the allowance is over and the
    # remedy is the account they have not made yet. Sending a trial user to a credits page for
    # an account they do not own is a dead end, so this branch comes FIRST.
    if _is_trial():
        if status in (402, 401, 403):
            return {"state": "trial_over",
                    "message": ("The starter allowance that came with OPYT is used up.")}
        # Indeterminate, and for a trial the two possibilities collapse: there is no balance
        # for this user to check, so there is nothing for them to do differently either way.
        return {"state": "unknown",
                "message": (f"Could not verify the starter allowance ({why}). It may have run "
                            f"out, or the network call may simply have failed. Calling "
                            f"`onboard` again re-checks; if it keeps saying this, the allowance "
                            f"is gone and the next step is a key of their own.")}
    if status == 402:
        return {"state": "unfunded",
                "message": (f"Your OpenRouter key works, but the account has no credit, so "
                            f"nothing can be embedded or classified. Add credit at {TOPUP_URL}, "
                            f"then call `onboard` again. Do not redo the approval — it would "
                            f"mint another key against the same empty balance. {COST_NOTE}")}
    if status in (401, 403):
        return {"state": "dead",
                "message": "Your stored OpenRouter key was rejected. Call `onboard` again to "
                           "approve a fresh one."}
    return {"state": "unknown",
            "message": (f"Could not verify the OpenRouter key ({why}). Two things cause this and "
                        f"we cannot tell them apart from here: the key is dead (call `onboard` "
                        f"again to mint a new one), or the account has no credit (add some at "
                        f"{TOPUP_URL}). Check the balance first — it is the cheaper test.")}


# ── the cheap trigger ─────────────────────────────────────────────────────────────────────────

# The two breakers that carry the model provider, named rather than discovered. A breaker whose
# name this module does not know is one it cannot interpret — `oracle-refresh:*` rows are open
# for a HANDLE's own trouble and mean nothing about the key, and `github.com` is not a model
# provider at all. Adding a third provider breaker means adding it here, deliberately.
_PROVIDER_BREAKERS = ("openrouter", "openrouter-embed")


def provider_blocked() -> str | None:
    """The name of a model-provider circuit that is not closed, or None. LOCAL — no network.

    This is the TRIGGER, never the answer. The breaker opens for any repeated provider failure —
    an OpenRouter outage trips it exactly as a spent allowance does — so what it buys is the
    right to spend ONE `openrouter()` probe at the moment something is actually wrong, instead
    of an HTTP round trip on every tool call to be told "no".

    ⚠️ IT READS THE RAW `state` COLUMN, AND `peek()` WOULD BE WRONG HERE. `peek` answers "would a
    call be permitted", which goes True as soon as the cooldown elapses — 60s for `openrouter` —
    so a dead key reads as permitted for all but the first minute after each trip. Measured on
    the live fixture 2026-09-15: the `openrouter` row sat OPEN with 14 consecutive
    `403 Key limit exceeded`, eight minutes past `opened_at`, where `peek()` answers True. The
    column is what stays true for as long as the trouble does, because only a `record_success`
    closes it and none can happen while the key is refused.

    Fail-safe in the quiet direction: an unreadable table means NOT blocked. Inventing an outage
    would put a top-up notice on a healthy answer, which is worse than the silence this fixes.
    """
    try:
        from pipeline.circuit_breaker import status
        for row in status():
            if row["service"] in _PROVIDER_BREAKERS and row["state"] != "closed":
                return str(row["service"])
    except Exception:
        return None
    return None
