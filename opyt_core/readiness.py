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
"""
from __future__ import annotations

TOPUP_URL = "https://openrouter.ai/credits"


def _credential(service: str) -> str | None:
    from pipeline.credentials import get_credential
    return get_credential(service)


def _ping(key: str) -> tuple[bool, str, int | None]:
    from pipeline.llm_client import validate_provider_status
    return validate_provider_status("openrouter", key)


def openrouter() -> dict:
    """{'state': missing|dead|unfunded|unknown|ok, 'message': str}. Never returns a key value."""
    if not _credential("openrouter"):
        return {"state": "missing",
                "message": "No OpenRouter key yet. `onboard` will open a browser tab — click "
                           "Approve and nothing needs pasting."}
    ok, why, status = _ping(_credential("openrouter"))
    if ok:
        return {"state": "ok", "message": "OpenRouter key is live and has credit."}
    if status == 402:
        return {"state": "unfunded",
                "message": (f"Your OpenRouter key works, but the account has no credit, so "
                            f"nothing can be embedded or classified. Add credit at {TOPUP_URL}, "
                            f"then call `onboard` again. Do not redo the approval — it would "
                            f"mint another key against the same empty balance.")}
    if status in (401, 403):
        return {"state": "dead",
                "message": "Your stored OpenRouter key was rejected. Call `onboard` again to "
                           "approve a fresh one."}
    return {"state": "unknown",
            "message": (f"Could not verify the OpenRouter key ({why}). Two things cause this and "
                        f"we cannot tell them apart from here: the key is dead (call `onboard` "
                        f"again to mint a new one), or the account has no credit (add some at "
                        f"{TOPUP_URL}). Check the balance first — it is the cheaper test.")}
