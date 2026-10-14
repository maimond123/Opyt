"""
pipeline/retrieval/served.py — ServedKB: the reads of a knowledge base served over HTTP.

A registered peer whose location is an https:// base URL is somebody's knowledge base served by
`service/app.py`. `ServedKB` answers search, open and aggregate for that peer with the same
envelope a file peer's store returns; the envelope crosses the wire verbatim. Every request
carries `as_kb`, the name this install registered the peer under, so the envelope comes back
naming a peer this install can resolve. Every transport failure becomes `peers.PeerUnavailable`.
"""
from __future__ import annotations

from opyt_core import service_http
from pipeline.retrieval import peers


def _call(row: dict, path: str, payload: dict | None = None, *, method: str = "post",
          params: dict | None = None) -> dict:
    """One HTTP exchange with the service, from a peer registry row.

    401 (token revoked, or the owner unpublished) and 404 (export not uploaded yet) get their own
    sentences naming the fix; any other non-2xx becomes `PeerUnavailable` with the cause."""
    url = f"{row['location']}/{path}"
    try:
        return service_http.request(method, url, token=row.get("token"), json=payload,
                                    params=params)
    except service_http.Unreachable as e:
        raise peers.PeerUnavailable(
            f"the service at {row['location']} could not be reached ({e}).") from e
    except service_http.Refused as e:
        if e.status == 401:
            raise peers.PeerUnavailable(
                "the service refused this install's token — it was revoked, or was never "
                "granted. Ask the owner for a new grant code and redeem it to restore "
                "access.") from e
        if e.status == 404:
            raise peers.PeerUnavailable(
                "this knowledge base has not arrived on the service yet — its owner shared it, "
                "and the copy usually lands within a minute or two of them doing so. Nothing is "
                "wrong with this install; try again shortly.") from e
        raise peers.PeerUnavailable(f"the service answered {e.status}: {e.detail}") from e


class ServedKB:
    """A knowledge base served by the sharing service, from its peer registry row. There is no
    served `feed`: the only caller of the feed is the service itself, reading its own exports."""

    def __init__(self, row: dict):
        self.row = row

    def search(self, query: str, *, k: int, **filters) -> dict:
        """POST the search and return the envelope verbatim. `filters` are the `SearchBody`
        field names, sent unnormalized."""
        return _call(self.row, "search", {"query": query, "k": k, "as_kb": self.row["name"],
                                          **filters})

    def open_atom(self, atom_id: str) -> dict:
        return _call(self.row, "open", {"atom_id": atom_id, "as_kb": self.row["name"]})

    def aggregate(self, scope: dict | None, sample: int = 0) -> dict:
        """The wire body has no `sample`, so a served aggregate carries no `corpus_sample`."""
        del sample
        return _call(self.row, "aggregate", {"scope": scope, "as_kb": self.row["name"]})

    def close(self) -> None:
        """Nothing held open between requests."""
