"""What every keyless paper-registry client (arXiv, OpenAlex) shares: one GET with the 429 split
(raising `SourceError`), the persisted per-host breaker, pacing, and the author cap.
"""

from __future__ import annotations

import threading
import time
import urllib.parse

import requests

from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError
from pipeline.host_access import request as host_request
from pipeline.ingestion.utils import log
from pipeline.platforms.errors import SourceError

_UA = "opyt/1.0 (+https://github.com/opyt)"
_TIMEOUT = 20.0

# How many of one paper's authors OPYT records, on a candidate's payload and on the atom
# (`ingest_papers.atom_authors` takes this same number). A bound on payload size for
# hyperauthored papers, not a judgement about who matters.
MAX_PAYLOAD_AUTHORS = 20


class RateLimited(RuntimeError):
    """The source told us to back off: stop asking, rather than try the next one."""


def get(url: str, *, headers: dict | None = None) -> bytes | None:
    """One GET: the body, or None when the host has no such resource (404). Raises `RateLimited`
    on 429 and lets every other failure raise, so the breaker running it counts the failure.

    A 404 or a timeout is about one request; a 429 is about us, so the host must stop being hit.
    """
    response = host_request(requests.get, url, headers={"User-Agent": _UA, **(headers or {})},
                            timeout=_TIMEOUT)
    if response.status_code == 429:
        raise RateLimited(f"HTTP 429 from {urllib.parse.urlsplit(url).netloc}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def call_get(breaker, url: str, *, headers: dict | None = None) -> bytes | None:
    """Run a GET through its host breaker: the body, or None when the host has no such resource.
    Raises `SourceError` when the breaker is open, the host throttles, or the request fails; the
    breaker has counted the failure first."""
    try:
        return breaker.call(lambda: get(url, headers=headers))
    except CircuitOpenError as exc:
        raise SourceError(f"paper registry breaker open ({exc}) — backing off") from None
    except RateLimited as exc:
        raise SourceError(str(exc)) from None
    except Exception as e:
        log(f"[paper-registry] GET failed {url[:110]}: {type(e).__name__}: {e}")
        raise SourceError(f"paper registry request failed: {type(e).__name__}: {e}") from None


# ── A persisted breaker, for the clients that own their own transport ───────────
class BreakerBacked:
    """One persisted circuit breaker per registry host, shared across client lifetimes.

    Fresh processes must honor prior failures and cooldowns. Clients use the same
    three-failure threshold and fifteen-minute cooldown; the host is their only setting.
    """
    breaker_host: str = ""

    def __init__(self, breaker=None):
        self._breaker = breaker

    def _get_breaker(self):
        if self._breaker is None:
            self._breaker = CircuitBreaker(self.breaker_host, threshold=3, cooldown=900.0)
        return self._breaker

    def available(self) -> bool:
        """False while the breaker is open, so a caller can skip this source without paying the
        politeness delay first.

        Uses `peek`, never `allow`: `allow` claims the half-open trial the real call needs.
        """
        try:
            return bool(self._get_breaker().peek())
        except Exception:
            return True          # a broken breaker must not silence a working source


# ── Pacing, for the paper path's own reads ─────────────────────────────────────

class Pace:
    """One host's spacing on the paper path: the earliest moment its next read may go."""
    def __init__(self):
        self.lock = threading.Lock()
        self.next_at = 0.0


def paced_read(adapter, pace: Pace, fn, label: str):
    """Run one read on the paper path: breaker-checked, paced, and fail-safe.

    `fn` takes the adapter and returns whatever it reads. Any failure — breaker open, rate limit,
    unparseable response — is None, because every caller runs inside `paper_from_url` or
    `classify_link_deep`, neither of which may raise.
    """
    # Before the delay: a host known to be down must not also cost a wait per caller.
    if not adapter.available():
        return None
    interval = float(getattr(adapter, "min_interval_s", 0.0) or 0.0)
    with pace.lock:
        wait = pace.next_at - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, interval))
        pace.next_at = time.monotonic() + interval
    try:
        return fn(adapter)
    except Exception as e:
        # Logged: None also means "unknown paper", so without this a spent budget is invisible.
        log(f"[papers] {label} read failed: {type(e).__name__}: {e}")
        return None
