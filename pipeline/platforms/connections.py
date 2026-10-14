"""Answer whether the user has connected X and Substack to OPYT, and whether Substack is ready.

Onboarding, oracle status, curation gates and the scrapers call `_connected_x` and
`_connected_substack` directly. Each asks the managed local session on demand; nothing caches an
answer.
"""

from pipeline.platforms.substack import local as substack_local
from pipeline.platforms.x import local as x_local


def _connected_x() -> bool:
    """Whether OPYT's own browser session is logged into X: a decrypt-free row-presence check of
    the profiles OPYT created locally."""
    return x_local.has_session()


def _connected_substack() -> bool:
    """Whether an OPYT-owned browser profile is logged into Substack.

    Not `read_substack_cookies`: this runs inside `onboarding.phase.derive`, so it scans only
    profiles OPYT created and decrypts nothing.
    """
    return substack_local.has_session()


def managed_substack_session_ready(*, attempts: int = 4, delay: float = 1.0) -> bool:
    """Whether the OPYT-managed Substack session answers an authenticated request right now, with
    a short retry to ride out the cookie flush after a login. Onboarding's "done" gate turns on it.

    True means safe to proceed: the session authenticated, or the read was refused for a reason
    that is not the user's sign-in state (Cloudflare/transport). False only when every attempt
    gives a clean signed-out answer.
    """
    return substack_local.session_ready(attempts=attempts, delay=delay)
