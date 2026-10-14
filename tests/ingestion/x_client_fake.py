"""A stand-in for `XClient` in tests: each request is answered by a function."""


class FakeXClient:
    """`get(op, query_id, variables, features, **kw)` answers each `graphql_get`; every
    `resolve_query_id` returns `query_id` and is recorded in `resolved`."""

    def __init__(self, get=None, *, query_id="qid", viewer=None):
        self.get = get or (lambda *_a, **_k: {})
        self.query_id = query_id
        self.viewer = viewer
        self.resolved: list[dict] = []

    def resolve_query_id(self, op, **kw):
        self.resolved.append({"op": op, **kw})
        return self.query_id

    def graphql_get(self, op, query_id, variables, features, **kw):
        return self.get(op, query_id, variables, features, **kw)

    def viewer_id(self):
        return self.viewer
