"""The model provider stopped working. Say so, on the call it just degraded.

⚠️ THE PRODUCT KNEW AT 05:08 AND WAITED TO BE ASKED. Measured on the live hosted home
2026-09-15: a starter allowance hit its ceiling, OpenRouter answered `403 Key limit exceeded`,
and four subsystems each degraded correctly and in private — `screen`'s classify skipped a batch
of 100 into a log, the embed circuit opened, `oracle_refresh` deferred its pairs as
`provider_down`, the sitting readers took their own 402 paths. The user found out the NEXT
MORNING, from symptoms, and reasonably diagnosed it as broken X auth. It was a missing cent.

The copy already existed and was good; the probe already existed and was right. They were wired
so that the only way to reach either was `readiness.openrouter()`, whose one caller is
`onboard_state.derive`, which is reached by calling `onboard` — a question a user screening
candidates has no reason to ask. Nothing here is a fifth component. It is the wire.

IT REPEATS, AND THAT IS THE DIFFERENCE FROM ITS NEIGHBOURS. `pull_runs.completion_notice` and
`new_material` both stamp and never fire again, because each announces an EVENT that is false on
the second telling. This announces a STATE that persists until the user acts. A notice that fired
once at 05:08 into a closed conversation and then went quiet would reproduce exactly the morning
described above.

The furniture rule is satisfied a different way. `completion_notice` must stamp because it rides
healthy answers; this one is gated on the call having ACTUALLY DEGRADED — classify skipped, a
pull deferred, a sitting refused — so it is never a rider on a good result. It is the explanation
for the thin answer the user is looking at right now, and an explanation that appears only when
there is something to explain cannot become wallpaper.

NO AMOUNT IS NAMED, and that is structural rather than careful. The sentence is rendered from
`readiness.openrouter()` rather than restated here, and readiness names no figure. A literal
"20 cents" would have been wrong for every key minted before 2026-09-15 — the live fixture above
holds a $0.10 ceiling against today's $0.20 default — and wrong again at the next raise. Text
that does not exist cannot go stale.

THE REMEDY COMES FROM READINESS TOO, because the three states want three different ones and
getting that mapping wrong is how a trial user gets sent to fund an account they never made.
"""
from __future__ import annotations

import time

from opyt_core import readiness

# The states worth interrupting a user for: the key is refused and only they can fix it.
#
# `unknown` IS DELIBERATELY ABSENT. It is the signature of a provider outage — the breaker is
# open because calls fail, and `readiness`'s own probe fails for the same reason, so it cannot
# tell a dead key from a dead network. Firing on it would tell a user with a perfectly funded
# account to go top it up, on the one day OpenRouter is having a bad time. Silence is the right
# failure here; the outage ends on its own and the key does not.
_ACTIONABLE = ("trial_over", "unfunded", "dead")

# Which call ends it. `unfunded` is absent because a funded-account holder's remedy is the
# top-up page `readiness` already names in its message, not another approval — minting a second
# key against the same empty balance is the loop this mapping exists to avoid. `dead` joins
# `trial_over` because a rejected key is replaced by approving a new one, not by paying.
_NEXT_CALL = {
    "trial_over": "onboard(start='openrouter')",
    "dead": "onboard(start='openrouter')",
}

# How long a verdict is trusted before the probe is paid for again. The notice repeats, so
# without this every degraded call would add an HTTP round trip to a call that is already
# failing slowly. A user who FIXES it does not wait this out: a working key closes the breaker on
# its first success, and a closed breaker returns None below without ever consulting the cache.
_VERDICT_TTL_S = 120.0

# ⚠️ A NON-ANSWER IS NOT AN ANSWER, AND MUST NOT BE TRUSTED LIKE ONE. `unknown` is what comes
# back when the probe could not run — and the usual reason it could not run is that the breaker
# the failing key just opened refuses it ("circuit 'openrouter' is OPEN — skipping call"). Held
# for the full TTL, that non-answer silenced the notice for two minutes: measured 2026-09-15, a
# revoked key produced `unknown` at t+0 and the first `dead` verdict only at t+120, so every
# degraded call in between went unexplained. Those are the FIRST calls the user makes after the
# key dies, which is precisely the window this module exists to cover.
#
# Short, not zero. When the breaker is open the re-probe costs nothing (it is refused locally,
# no socket), but `unknown` also covers a real network failure, and retrying a timeout on every
# degraded call would add that latency to answers that are already slow. 15s keeps the retry
# cheap while collapsing the blind spot to the breaker's own cooldown, which is irreducible.
_UNKNOWN_TTL_S = 15.0

_CACHE: tuple[float, dict] | None = None


def _verdict() -> dict:
    """`readiness.openrouter()`, at most once per TTL. See `_VERDICT_TTL_S`/`_UNKNOWN_TTL_S`."""
    global _CACHE
    now = time.time()
    if _CACHE is not None and now < _CACHE[0]:
        return _CACHE[1]
    verdict = readiness.openrouter()
    ttl = _UNKNOWN_TTL_S if verdict.get("state") == "unknown" else _VERDICT_TTL_S
    _CACHE = (now + ttl, verdict)
    return verdict


def allowance_notice() -> dict | None:
    """Why this call came back thin, when the reason is the model provider — or None.

    Cheap durable trigger, then one authoritative probe: `readiness.provider_blocked()` is a
    local SQLite read that answers "no" for free on every healthy call, and only a breaker that
    is genuinely not closed buys the network round trip that can tell an exhausted allowance
    apart from an outage.

    Fail-safe and silent throughout. This is an explanation attached to somebody else's answer
    and it must never be able to break it, so every failure path returns None — the cost is a
    thin result with no reason given, which is the status quo, rather than a tool call that dies
    while trying to be helpful.
    """
    try:
        blocked = readiness.provider_blocked()
        if blocked is None:
            return None
        verdict = _verdict()
        state = verdict.get("state")
        if state not in _ACTIONABLE:
            return None
    except Exception:
        return None

    notice: dict = {
        "code": "model_provider_blocked",
        "state": state,
        "circuit": blocked,
        # RENDERED, never restated — `readiness` owns what is wrong and what the person does
        # about it, and a second copy here would be the one that goes stale.
        "reason": verdict.get("message", ""),
    }
    if (call := _NEXT_CALL.get(state)) is not None:
        notice["next_call"] = call
    else:
        notice["top_up_url"] = readiness.TOPUP_URL

    notice["host_note"] = (
        "⚠️ SAY THIS NOW, BEFORE the result itself — this answer is thin BECAUSE of it, and a "
        "user who is not told will diagnose it as something else entirely (the last one blamed "
        "their X login, and so did the assistant helping them). Lead with the `reason` above in "
        "your own words, then give them what did come back. Everything already collected is "
        "still there and still searchable — say that, because 'the model provider is refused' "
        "sounds like data loss and it is not. Do NOT present this as a bug, an outage or "
        "something you will retry: it needs one action from them and nothing changes until they "
        "take it. If you have already told them this earlier in the conversation, one short "
        "clause is enough — do not repeat the whole explanation.")
    return notice
