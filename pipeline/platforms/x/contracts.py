"""The raw X transport's contract, and the values that cross it."""

from typing import NamedTuple, Protocol, runtime_checkable

HOME_URL = "https://x.com/home"
GRAPHQL_ROOT = "https://x.com/i/api/graphql"

# Public web-app bearer — not a secret; the constant the x.com web client ships.
FALLBACK_BEARER = ("AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
                   "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA")


class XRequest(NamedTuple):
    """One GraphQL operation, as the transport sends it. Most operations are GETs with their
    arguments in the query string; a few (SearchTimeline) are POSTs with them in a JSON body."""
    op: str
    query_id: str
    variables: dict
    features: dict
    field_toggles: dict | None = None
    method: str = "GET"


class XResponse(NamedTuple):
    """One raw response, before any policy reads it.

    `status` is the HTTP status. `rate_limit` is `(remaining, reset)` when the response carried
    both. `error` is an HTTP or decode failure, held rather than raised so the client records rate
    evidence first. `text` is the body, for diagnostics.
    """
    status: int
    data: dict | None
    rate_limit: tuple[int, float] | None = None
    error: Exception | None = None
    text: str = ""


@runtime_checkable
class XTransport(Protocol):
    """Where one X request runs: the managed local session."""

    def request(self, request: XRequest) -> XResponse:
        """Send one operation and return its raw response."""

    def discover_query_id(self, op: str, page_url: str) -> str | None:
        """Find an operation's identifier in X's web bundles, or None."""

    def viewer_id(self) -> str | None:
        """The signed-in account's id when this transport can establish it."""

    def refusal(self, action: str, op: str, response: XResponse | None) -> Exception:
        """The exception for a refused request, worded for this transport; it carries the
        response body."""
