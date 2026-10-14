"""The one X request path: query-ID, budget and response policy over the transport."""

from typing import Literal, TypeAlias

import pipeline.platforms.x.query_ids as query_ids
import pipeline.platforms.x.rate as x_rate
from pipeline.ingestion.utils import log
from pipeline.platforms.x.contracts import HOME_URL, XRequest, XTransport

ResponseAction: TypeAlias = Literal[
    "rate_limited", "unauthenticated", "drift", "http_error", "graphql_errors", "tolerated_errors", "return",
]


def response_action(status: int, *, has_errors: bool = False,
                    tolerate_errors: bool = False) -> ResponseAction:
    """Classify an HTTP status without performing side effects."""
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "unauthenticated"
    if status in (400, 404):
        return "drift"
    if not 200 <= status < 300:
        return "http_error"
    if has_errors:
        return "tolerated_errors" if tolerate_errors else "graphql_errors"
    return "return"


class XClient:
    """One X reader for a call: every endpoint reads through it.

    The policy every request shares lives here once: env → seed → cache → discovery for the
    operation id, a local refusal of a spent budget, rate evidence recorded off every response
    (429 included, before any refusal), and the response's classification. The transport only
    sends; its `refusal` words the exception.
    """

    def __init__(self, transport: XTransport) -> None:
        self.transport = transport

    def resolve_query_id(self, op: str, *, default_seed: str = "", env_var: str | None = None,
                         page_url: str = HOME_URL) -> str:
        query_id = query_ids.resolve(
            op, default_seed=default_seed, env_var=env_var,
            discover=lambda: self.transport.discover_query_id(op, page_url))
        if query_id is None:
            raise self.transport.refusal("missing_query_id", op, None)
        return query_id

    def graphql_get(self, op: str, query_id: str, variables: dict, features: dict, *,
                    field_toggles: dict | None = None, tolerate_errors: bool = False) -> dict:
        """One GraphQL GET, under `_send`'s policy."""
        return self._send(XRequest(op, query_id, variables, features, field_toggles),
                          tolerate_errors=tolerate_errors)

    def graphql_post(self, op: str, query_id: str, variables: dict, features: dict, *,
                     tolerate_errors: bool = False) -> dict:
        """One GraphQL POST (an operation the web client posts, such as SearchTimeline), under
        the same policy as a GET."""
        return self._send(XRequest(op, query_id, variables, features, method="POST"),
                          tolerate_errors=tolerate_errors)

    def _send(self, request: XRequest, *, tolerate_errors: bool) -> dict:
        """The policy every request shares. On 401/403 -> SyncAuthError (dead session). On
        400/404 -> clear the queryId cache (queryId/features drifted) and raise so the next run
        re-discovers. `tolerate_errors`: X often returns a partial `errors` array alongside valid
        `data`; when True, keep `data` and just log the error count instead of hard-failing.

        Budgeted per operation off x.com's own headers — see `x_rate.refuse_if_spent`."""
        op = request.op
        x_rate.refuse_if_spent(op)
        response = self.transport.request(request)
        # Recorded on every status: a 429 is the response whose reset instant we most need.
        if response.rate_limit is not None:
            x_rate.record_rate_values(op, *response.rate_limit)
        action = response_action(response.status, tolerate_errors=tolerate_errors)
        if action == "return":
            if response.error is not None:
                raise response.error                      # a 2xx whose body did not decode
            data = response.data or {}
            action = response_action(response.status, has_errors=bool(data.get("errors")),
                                     tolerate_errors=tolerate_errors)
            if action == "return":
                return data
            if action == "tolerated_errors":
                log(f"[x-graphql] {op}: {len(data['errors'])} partial field error(s) tolerated "
                    f"(data present)")
                return data
        if action == "drift":
            query_ids.qid_cache_clear(op)
        raise self.transport.refusal(action, op, response)

    def viewer_id(self) -> str | None:
        return self.transport.viewer_id()
