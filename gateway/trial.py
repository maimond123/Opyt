"""
gateway/trial.py — the starter allowance: minting it, and the ledger that makes it one per person.

WHAT THIS CHANGES ABOUT THE GATEWAY, STATED PLAINLY. `gateway/app.py` used to be able to say it
held no user key and no durable state. Both stop being true here, and neither should be discovered
by a reader who trusted that docstring:

  • A MANAGEMENT KEY lives in this process's environment. It is strictly more dangerous than any
    user key the gateway has ever handled, because it can mint spend against the operator's own
    OpenRouter balance. It never enters a child's environment (`_GATEWAY_ONLY` in `children.py`),
    never appears in a response body, and never reaches a log line.
  • A LEDGER is written to disk. Everything else the gateway keeps is in memory on purpose —
    a persisted row about a running child would outlive the process it describes. This row
    describes something that OUTLIVES the process by design: a person has had their allowance,
    and that fact must survive a deploy. If it did not, every restart would re-open the faucet.

ONE PER SUBJECT, NOT PER INSTALL. A subject is a Google `sub`, the same claim that names a hosted
home. Keying on anything the client controls — an install id, a machine fingerprint — makes
reinstalling the way to get another allowance, which is to say it makes the cap decorative.

WHY THE MINT IS CHEAP EVEN WHEN IT IS WASTED. `limit` is a CEILING, not a prepayment: OpenRouter
bills actual usage. A key minted for somebody who never returns costs nothing, which is what makes
minting eagerly — before the user has shown any intent — the right call rather than a gamble.

WHAT IS DELIBERATELY NOT HERE. No `limit_reset`. A daily-resetting cap would turn every trial key
into a permanent free tier against the operator's balance, which is a different product and a much
more expensive one.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

KEYS_ENDPOINT = "https://openrouter.ai/api/v1/keys"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"

MANAGEMENT_KEY_ENV = "OPYT_TRIAL_MANAGEMENT_KEY"

# Buys a first corpus and the moment of seeing it work. The upgrade to the user's own account is
# a step of the design, not a failure of it.
#
# The failure mode is asymmetric, which is what sets the number. Too HIGH costs the operator money
# he is billed for only if it is spent; too LOW strands a user part-way through their first corpus,
# which is worse than never offering a trial — they meet `trial_over` before they have seen the
# thing work, so the goodwill the allowance was bought to create never happens.
#
# MEASURED 2026-09-15 via `scripts/trial_keys.py`, on 3 real keys that had all been used:
#
#     heaviest   $0.0149    a complete hosted onboarding — 7 Oracles across X and Substack,
#                           1,026 atoms, bookmarks and saved posts imported
#     mean       $0.0067
#     median     $0.0030
#
# ⚠️ RE-MEASURED 2026-09-16, AND THE HEADROOM ABOVE IS GONE. One hosted onboarding spent $0.2001
# — the full 0.20 ceiling — in 36 minutes, from a key minted at connect. That is 13.4x the
# "heaviest complete corpus ever built" the day-old figure above describes, which is to say the
# day-old figure no longer describes this product. What changed in between: the background rails
# were fixed (`dac2d535`, `44b0280d`, `6654a7ca`), so `curation_catchup` and `candidate_probe`
# now RUN — during the 09-15 measurement neither ever started — and `ca53d51a` began answering
# "what is my corpus about" by reading the corpus with a model instead of labelling it. Both add
# real spend and neither is in the number above. The 09-15 sample is kept rather than deleted
# because the CONTRAST is the finding; read it as history, not as a current cost.
#
# So the asymmetry that set this number now cuts the other way. At 0.20 the allowance ends at
# almost exactly the end of a first corpus, which is the one user-visible failure it exists to
# prevent — and a corpus larger than the one measured meets it MID-build.
#
# 0.25 is David's call, taken 2026-09-16 with the 13.4x above in front of him. State plainly what
# it buys: about 25% headroom over a single measured onboarding, not the 13x the previous comment
# described. It is defensible because running out is a DESIGNED moment and not a dead end —
# `pipeline/kb/allowance_notice.py` fires on the degraded call and `_trial_over_prompt` hands the
# user to their own OpenRouter account — but it is thin, and one heavier-than-measured corpus
# will meet it early. n is still ONE run under current code. Re-run `trial_keys.py` after real
# signups and set this from a distribution rather than from a single observation.
#
# What bounds the downside is `DEFAULT_DAILY_CAP`, not this number — see it below.
DEFAULT_LIMIT_USD = 0.25
DEFAULT_TTL_DAYS = 30
# A blast radius, not a business rule: if something is minting in a loop, this is what stops it
# before the balance does. Tune it to real signups, not to what feels generous.
DEFAULT_DAILY_CAP = 200

_PENDING_TTL_SECONDS = 600.0


class TrialUnavailable(RuntimeError):
    """Refused. The reason is a short word the client turns into "no allowance available" —
    never into a retry, because nothing the client can do changes any of these answers.

    `detail` is for the OPERATOR and never crosses the wire: the short word is what the child
    receives, and the provider's own sentence is what gets logged. Without this split an
    upstream refusal reached the gateway log as the bare word "upstream", which is what made
    two separate 400s — a bad `expires_at` spelling, then `creator_user_id` on a personal
    account — each cost a round of guessing. A refusal nobody can read is a refusal nobody can
    fix.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.detail = detail


def enabled() -> bool:
    """Whether this gateway can mint at all. An operator who runs OPYT without a management key
    gets a gateway that routes OAuth exactly as before and advertises no trial — which is the
    fail-safe shape: the absent optional input degrades to the old flow, not to an error."""
    return bool(os.environ.get(MANAGEMENT_KEY_ENV))


def _config() -> tuple[float, int, int]:
    return (float(os.environ.get("OPYT_TRIAL_LIMIT_USD", DEFAULT_LIMIT_USD)),
            int(os.environ.get("OPYT_TRIAL_TTL_DAYS", DEFAULT_TTL_DAYS)),
            int(os.environ.get("OPYT_TRIAL_DAILY_CAP", DEFAULT_DAILY_CAP)))


# ── the ledger ────────────────────────────────────────────────────────────────────────────────

class Ledger:
    """subject → the allowance they already had. The one durable thing the gateway writes.

    A plain JSON file under one in-process lock, which is sound for exactly the reason the app
    docstring gives for `uvicorn --workers 1`: this runs single-process on purpose. If that ever
    stops being true, this file is the first thing that breaks, and it should be moved to the
    same store the workers share rather than be given a lock that spans processes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A ledger that cannot be read is NOT treated as empty — that would be the one
            # failure mode where the cap silently disappears. `claim` raises on a read it cannot
            # trust, so the gateway refuses to mint rather than over-minting.
            if self.path.exists():
                raise TrialUnavailable("ledger_unreadable")
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)        # atomic: a torn ledger is an over-minting ledger

    def claimed(self, subject: str) -> dict | None:
        with self._lock:
            return self._read().get(subject)

    def reserve(self, subject: str, daily_cap: int) -> None:
        """Prove this subject may mint, and hold their slot before any money is spent.

        Reserve-then-fill rather than mint-then-record: two calls that arrive together must not
        both pass the check, and a mint that fails after this leaves a spent slot rather than a
        second key. A spent slot costs one person one allowance; a double mint costs the
        operator every time it happens.
        """
        with self._lock:
            data = self._read()
            if subject in data:
                raise TrialUnavailable("already_claimed")
            cutoff = time.time() - 86400
            if sum(1 for row in data.values() if row.get("at", 0) > cutoff) >= daily_cap:
                raise TrialUnavailable("daily_cap")
            data[subject] = {"at": time.time(), "hash": None}
            self._write(data)

    def fill(self, subject: str, key_hash: str, limit: float, expires_at: str) -> None:
        with self._lock:
            data = self._read()
            row = data.get(subject) or {"at": time.time()}
            row.update(hash=key_hash, limit=limit, expires_at=expires_at)
            data[subject] = row
            self._write(data)

    def rows(self) -> dict[str, dict]:
        """Every subject this gateway has minted for. Read-only, for the operator's audit."""
        with self._lock:
            return dict(self._read())

    def release(self, subject: str) -> None:
        """Give the slot back when the mint never happened. Only ever called on an upstream
        failure — never on a refusal, because a refusal means they had their turn."""
        with self._lock:
            data = self._read()
            if (data.get(subject) or {}).get("hash") is None:
                data.pop(subject, None)
                self._write(data)


# ── minting ───────────────────────────────────────────────────────────────────────────────────

async def mint(http: httpx.AsyncClient, subject: str, ledger: Ledger) -> dict:
    """Create one capped, expiring OpenRouter key for `subject`. Returns the plaintext ONCE.

    OpenRouter returns the key value in this response and never again, which is why nothing here
    retries a mint it is unsure about: a second attempt after an ambiguous failure is a second
    key against the same balance that nobody can ever read back to revoke.
    """
    if not enabled():
        raise TrialUnavailable("not_configured")
    limit, ttl_days, daily_cap = _config()
    ledger.reserve(subject, daily_cap)
    # ⚠️ `Z`, NOT `+00:00`. Python's `isoformat()` emits the offset form and OpenRouter rejects
    # it outright — `400 Invalid request field: expires_at` — while accepting the identical
    # instant written with a `Z`. Found live on 2026-09-11, after the first real mint failed and
    # every hosted user silently fell through to the OpenRouter approval this exists to avoid.
    # The docs list `expires_at` as a valid field and say nothing about which spelling; only the
    # API knows. `_state_of` reads these back with `fromisoformat`, which accepts `Z` on 3.11+.
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    try:
        response = await http.post(
            KEYS_ENDPOINT,
            headers={"Authorization": f"Bearer {os.environ[MANAGEMENT_KEY_ENV]}"},
            json={
                # The subject lives in the NAME, and that is the only attribution this mint
                # carries. `creator_user_id` was sent here until 2026-09-11 and OpenRouter
                # answers `400 creator_user_id is only valid for organization-owned keys` — it
                # is an organization feature, and a personal account is what most operators
                # will run. The real record of who has had an allowance is the ledger, which is
                # ours and does not depend on the provider's account type at all.
                "name": f"opyt-trial-{subject}",
                "limit": limit,
                "expires_at": expires_at,
                # NOT `limit_reset`. See the module docstring — a reset makes this a free tier.
            },
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
    except httpx.HTTPError as e:
        ledger.release(subject)
        raise TrialUnavailable("upstream", f"{type(e).__name__}: {e}") from None
    if response.status_code >= 300:
        ledger.release(subject)
        raise TrialUnavailable("upstream",
                               f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    key = body.get("key")
    if not key:
        ledger.release(subject)
        raise TrialUnavailable("upstream", f"no key in response: {str(body)[:300]}")
    key_hash = str((body.get("data") or {}).get("hash") or "")
    ledger.fill(subject, key_hash, limit, expires_at)
    return {"key": key, "hash": key_hash, "limit": limit, "expires_at": expires_at}


# ── the local flow's two short-lived stores ───────────────────────────────────────────────────

@dataclass
class _Pending:
    value: dict
    expires_at: float


class PendingStore:
    """One-use, short-lived, in memory — the same shape as the interaction nonces beside it, and
    for the same reason: these describe a browser round trip that is either finished within
    minutes or abandoned. Nothing here is worth surviving a restart, and a minted key sitting in
    a file waiting to be collected would be."""

    def __init__(self) -> None:
        self._items: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def put(self, value: dict, *, ttl: float = _PENDING_TTL_SECONDS) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.monotonic()
            self._items = {k: v for k, v in self._items.items() if v.expires_at > now}
            self._items[token] = _Pending(value=value, expires_at=now + ttl)
        return token

    def take(self, token: str) -> dict | None:
        with self._lock:
            item = self._items.pop(token, None)
        if item is None or item.expires_at <= time.monotonic():
            return None
        return item.value


def loopback_callback(url: str) -> str | None:
    """The callback a local install asked us to redirect to, or None if it is not its own.

    THIS IS THE OPEN-REDIRECT BOUNDARY. The gateway is about to bounce a browser to a URL the
    caller supplied, and one turn later that URL receives a code that buys a funded key. Only a
    loopback HTTP address on this user's own machine can be that target, so the check is an
    allow-list of hosts rather than a scan for anything suspicious.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        return None
    if parsed.query or parsed.fragment:
        # The code is appended as the ONLY query parameter, so a callback that arrives carrying
        # its own is either a mistake or an attempt to smuggle one past the parse on the far end.
        return None
    return url


def verify_pkce(verifier: str, challenge: str) -> bool:
    import hashlib
    got = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return secrets.compare_digest(got, challenge)


def subject_from_id_token(id_token: str) -> str | None:
    """The `sub` claim out of an id_token that came STRAIGHT from Google's token endpoint.

    No signature check, and that is correct here rather than lazy: OpenID Connect Core 3.1.3.7
    excuses it for exactly this case — the token arrived over TLS, from the issuer, in the
    response to a request authenticated with our own client secret. There is no untrusted hop
    for a forged token to enter through. A token reaching us any OTHER way would need full
    verification, which is why this function names the one source it is for.
    """
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        sub = json.loads(base64.urlsafe_b64decode(payload)).get("sub")
    except (IndexError, ValueError, TypeError):
        return None
    return str(sub) if sub else None


# ── the operator's audit: what the allowance actually costs, and what to switch off ────────────
# ONE call answers both questions, because OpenRouter is the party doing the billing. That is
# also why this is not the dollar accounting `retired-pipeline-llm-spend` bans: nothing here
# prices a token or keeps a ledger of spend. It ASKS the provider what a key has used, on demand,
# on an operator's box, and the pipeline neither imports it nor knows it exists.
#
# `usage` is the number that settles `readiness.COST_NOTE`. That string promises a user their
# reading costs a few pennies, and until a real corpus has been built against a real key it is an
# intention rather than a measurement. This is how it stops being one.

async def audit(http: httpx.AsyncClient, ledger: Ledger) -> list[dict]:
    """Every minted key's live state, straight from OpenRouter. Never returns a key value.

    A row the provider no longer knows about comes back `missing` rather than being dropped: a
    hash in the ledger with nothing behind it is worth seeing, because it means a mint was
    recorded that OpenRouter did not keep.
    """
    if not enabled():
        raise TrialUnavailable("not_configured")
    headers = {"Authorization": f"Bearer {os.environ[MANAGEMENT_KEY_ENV]}"}
    out = []
    for subject, row in sorted(ledger.rows().items()):
        key_hash = row.get("hash")
        entry = {"subject": subject, "hash": key_hash, "minted_at": row.get("at"),
                 "limit": row.get("limit"), "expires_at": row.get("expires_at")}
        if not key_hash:
            # Reserved and never filled — a mint that failed after taking its slot.
            out.append({**entry, "state": "unfilled", "usage": None})
            continue
        try:
            response = await http.get(f"{KEYS_ENDPOINT}/{key_hash}", headers=headers,
                                      timeout=httpx.Timeout(connect=5.0, read=20.0,
                                                            write=5.0, pool=5.0))
        except httpx.HTTPError:
            out.append({**entry, "state": "unreachable", "usage": None})
            continue
        if response.status_code == 404:
            out.append({**entry, "state": "missing", "usage": None})
            continue
        if response.status_code in (401, 403):
            # Kept APART from `unreachable`, because the remedies are opposite and an operator
            # reading "unreachable" on every row concludes OpenRouter is down when in fact their
            # own management key is wrong. It is reported per row and the caller stops on it.
            out.append({**entry, "state": "unauthorized", "usage": None})
            continue
        if response.status_code >= 300:
            out.append({**entry, "state": "unreachable", "usage": None})
            continue
        data = (response.json() or {}).get("data") or {}
        out.append({**entry,
                    "usage": data.get("usage"),
                    "limit_remaining": data.get("limit_remaining"),
                    "disabled": bool(data.get("disabled")),
                    "state": _state_of(data, row)})
    return out


def _state_of(data: dict, row: dict) -> str:
    """`live`, `spent`, `expired` or `off` — what the operator would do about this key.

    Spent and expired are reported apart even though a user meets both as one `trial_over`,
    because they mean different things HERE: spent is the allowance working as designed, and a
    pile of expired-unspent keys means the allowance is being claimed and never used, which is a
    product signal rather than a cost.
    """
    if data.get("disabled"):
        return "off"
    remaining = data.get("limit_remaining")
    if remaining is not None and remaining <= 0:
        return "spent"
    expires_at = row.get("expires_at") or ""
    try:
        if expires_at and datetime.fromisoformat(expires_at) <= datetime.now(timezone.utc):
            return "expired"
    except ValueError:
        pass
    return "live"


async def disable(http: httpx.AsyncClient, key_hash: str) -> bool:
    """Switch one key off for good. Used only on a key that can no longer spend anyway — this is
    housekeeping so the provider's key listing stays readable, never a way to cut somebody off
    mid-use. `disabled` rather than DELETE, so the row survives to be audited."""
    if not enabled():
        raise TrialUnavailable("not_configured")
    try:
        response = await http.patch(
            f"{KEYS_ENDPOINT}/{key_hash}",
            headers={"Authorization": f"Bearer {os.environ[MANAGEMENT_KEY_ENV]}"},
            json={"disabled": True},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
    except httpx.HTTPError:
        return False
    return response.status_code < 300


def cost_summary(rows: list[dict]) -> dict:
    """What the allowance costs per person who actually used it.

    Keys with NO usage are counted separately rather than averaged in. A mint that was never used
    costs nothing — `limit` is a ceiling, not a prepayment — so folding those zeros into the mean
    would understate what a real user costs by however many tyre-kickers signed up, which is
    exactly the wrong direction for a number that backs a promise to users.
    """
    used = sorted(r["usage"] for r in rows
                  if isinstance(r.get("usage"), (int, float)) and r["usage"] > 0)
    unused = sum(1 for r in rows
                 if isinstance(r.get("usage"), (int, float)) and not r["usage"])
    if not used:
        return {"keys": len(rows), "used": 0, "unused": unused, "total_usd": 0.0,
                "mean_usd": None, "median_usd": None, "max_usd": None}
    return {
        "keys": len(rows),
        "used": len(used),
        "unused": unused,
        "total_usd": round(sum(used), 4),
        "mean_usd": round(sum(used) / len(used), 4),
        "median_usd": round(used[len(used) // 2], 4),
        "max_usd": round(used[-1], 4),
    }
