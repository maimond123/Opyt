"""
service/google_signin.py — the connector's Google sign-in: FastMCP's `GoogleProvider`, with token
revocation (RFC 7009) that ends the sign-in.

`GoogleProvider` offers no `/revoke`. Here a client that disconnects can revoke either of its
tokens, and the whole sign-in ends: the Google grant stored behind the token is deleted, so every
token issued from it is refused from the next request, and Google is asked to revoke the grant too.
A user who removes Opyt in their Google account needs none of this: every call checks its Google
token with Google first, so a grant Google revoked is refused on the next call.

`/revoke` is this module's own handler, because the MCP SDK's revokes nothing here (measured on the
SDK and FastMCP the service box runs and on the current ones). It refuses a request without a
`client_secret`, which the public clients registered here never have. It revokes an access token
only when the token's `client_id` is the client asking, and `GoogleProvider` puts the Google
account there. And FastMCP's revocation sends Google the token it was handed, which is FastMCP's
own, not Google's.
"""
from __future__ import annotations

import httpx
from fastmcp.server.auth.providers.google import GoogleProvider
from mcp.server.auth.provider import RefreshToken
from mcp.server.auth.routes import cors_middleware
from mcp.server.auth.settings import RevocationOptions
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
REVOKE_PATH = "/revoke"
_TIMEOUT = 15.0


async def revoke_at_google(token: str) -> None:
    """Ask Google to revoke the grant `token` belongs to. A failure is not raised: the grant is
    already deleted here, so the client is refused either way."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(GOOGLE_REVOKE_URL, data={"token": token})
    except httpx.HTTPError:
        pass


class GoogleSignIn(GoogleProvider):
    """`GoogleProvider` whose `/revoke` ends the sign-in a token came from."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Advertises `revocation_endpoint` in the server's metadata; the route is replaced below.
        self.revocation_options = RevocationOptions(enabled=True)

    def get_routes(self, mcp_path: str | None = None) -> list:
        routes = [r for r in super().get_routes(mcp_path)
                  if getattr(r, "path", None) != REVOKE_PATH]
        return [*routes, Route(REVOKE_PATH, cors_middleware(self._revoke, ["POST", "OPTIONS"]),
                               methods=["POST", "OPTIONS"])]

    async def _revoke(self, request: Request) -> Response:
        """RFC 7009: `token` and `client_id`, no secret. Always 200, whether or not the token
        was one this server issued to that client (§2.2)."""
        form = await request.form()
        token, client_id = str(form.get("token") or ""), str(form.get("client_id") or "")
        issued = self._issued(token)
        if issued is not None and issued[1].get("client_id") == client_id:
            await self._end_sign_in(*issued, token)
        return Response(status_code=200,
                        headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    def _issued(self, token: str) -> tuple[str, dict] | None:
        """`(kind, claims)` of a token this server signed and that has not expired, or None."""
        for kind in ("access", "refresh"):
            try:
                return kind, self.jwt_issuer.verify_token(token, expected_token_use=kind)
            except Exception:      # not ours, the other kind, or expired: each just "not this"
                continue
        return None

    async def _end_sign_in(self, kind: str, claims: dict, token: str) -> None:
        """Delete the Google grant behind a token, which every token issued from it reads through,
        then have Google revoke it too."""
        if kind == "refresh":
            # FastMCP's own record of the refresh token (its revocation does only this here).
            await super().revoke_token(RefreshToken(token=token, client_id=claims["client_id"],
                                                    scopes=[]))
        mapping = await self._jti_mapping_store.get(key=claims["jti"])
        grant = (await self._upstream_token_store.get(key=mapping.upstream_token_id)
                 if mapping is not None else None)
        if grant is None:
            return
        await self._upstream_token_store.delete(key=grant.upstream_token_id)
        await revoke_at_google(grant.refresh_token or grant.access_token)
