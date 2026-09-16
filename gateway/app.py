"""
gateway/app.py — the public HTTPS endpoint claude.ai points at.

From claude.ai's side this IS the Opyt MCP server. It is not: it authenticates the caller,
resolves them to a `sub` claim, and reverse-proxies the request to that user's own `opyt-mcp`
child on loopback. Design record: docs/plans/2026-09-02-hosted-opyt-remote-connector.md.

**Why a gateway rather than one multi-tenant server.** A child serves one `$OPYT_HOME`, so its
module-level globals stay per-user without auditing a line of tool code. See `children.py`.

**OAuth, and why Google is not the authorization server.** claude.ai requires dynamic client
registration from an MCP server's authorization server, and Google does not offer it. So this
process plays authorization server toward claude.ai and delegates the human login to Google —
`GoogleProvider` is an OAuth proxy that does exactly that. There are two authorization codes:
claude.ai gets one minted here, Google's never leaves this process. The bearer token claude.ai
stores is ours, so the upstream provider can be swapped without claude.ai noticing.

**Run it single-process.** `uvicorn --workers N` would give each worker its own routing table
and spawn N children per user, all writing one home. There is nothing to parallelize here —
every route is I/O — so `python -m gateway` runs one process on purpose.

**What this process stores:** running children and short-lived interaction nonces, both in memory
(see `children.py` for why), plus whatever the auth provider keeps for tokens. It holds no
onboarding state — `pipeline/kb/onboard_state.derive()` recomputes that from disk inside each
child on every call, and a committed guard bans a phase file — no cookies and no corpus.

**Two of those claims changed on 2026-09-11, when the starter allowance arrived.** This file
used to end "and no user key", and everything it kept was in memory. Both are now qualified, and
the qualification is the point rather than a footnote — see `gateway/trial.py`:

  • a MANAGEMENT KEY sits in this process's environment, able to mint spend against the
    operator's own OpenRouter balance. It is stripped from every child, never returned in a
    response, and never logged. It is the most dangerous thing this process has ever held.
  • a LEDGER of who has had their allowance is written to DISK, because that fact has to
    outlive a deploy. If it did not, every restart would hand everyone a fresh allowance.

Minted keys themselves are not stored: the plaintext passes through to the install that asked
for it and is held nowhere, which is also the only way it can be held — OpenRouter returns it
exactly once.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastmcp.server.auth.providers.google import GoogleProvider
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (HTMLResponse, JSONResponse, RedirectResponse, Response,
                                 StreamingResponse)
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from gateway import trial
from gateway.children import (BROWSER_LOGIN_KINDS, CHILD_LOG, DEFAULT_MAX_SIGNINS,
                              BadSubject, ChildPool, InteractionNonce, SpawnFailed)
from pipeline.ingestion import hosted_browser
from opyt_core import web_panel

MCP_PATH = "/mcp"
_STATIC_ROOT = Path(__file__).with_name("static")

# RFC 2616 hop-by-hop headers: they describe one connection and must not be relayed onto
# another. `content-length` is dropped on the way UP because httpx recomputes it, and kept on
# the way DOWN because `aiter_raw` relays the body exactly as the child framed it.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})
_DROP_UPSTREAM = _HOP_BY_HOP | {"host", "authorization", "content-length"}


def _log(message: str) -> None:
    """One operator line on the gateway's own stderr, which systemd captures as its journal.

    Deliberately not `pipeline.ingestion.utils.log`: this package never imports `pipeline`, so
    that the proxy half cannot reach the half that handles user data. Deliberately not the
    `logging` module either — uvicorn configures handlers only under its own logger names, so
    an unconfigured logger would drop exactly the failures this exists to record.
    """
    print(f"[gateway] {message}", file=sys.stderr, flush=True)


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"{name} is required to run the gateway")
    return val


def build_app(*, base_url: str | None = None,
              client_id: str | None = None,
              client_secret: str | None = None,
              homes_root: Path | str | None = None,
              idle_seconds: float | None = None) -> Starlette:
    """The ASGI app. Arguments override the environment, which is how tests build one."""
    base_url = base_url or _env("OPYT_GATEWAY_BASE_URL")
    client_id = client_id or _env("OPYT_GATEWAY_GOOGLE_CLIENT_ID")
    client_secret = client_secret or _env("OPYT_GATEWAY_GOOGLE_CLIENT_SECRET")
    root = Path(homes_root or os.environ.get("OPYT_HOMES_ROOT")
                or Path.home() / ".opyt-homes")
    idle = float(idle_seconds if idle_seconds is not None
                 else os.environ.get("OPYT_GATEWAY_IDLE_SECONDS", 900))
    # A garbage value raises here rather than at the first sign-in, which is the same fail-loud
    # posture `_env` takes: a cap the operator cannot read back is worse than no cap.
    max_signins = int(os.environ.get("OPYT_GATEWAY_MAX_SIGNINS", DEFAULT_MAX_SIGNINS))
    internal_url = os.environ.get(
        "OPYT_GATEWAY_INTERNAL_URL",
        f"http://127.0.0.1:{os.environ.get('OPYT_GATEWAY_PORT', '8080')}",
    )

    # `openid` and `email` are what make Google's tokeninfo return `sub`, and `sub` is the only
    # claim this design uses: it names the home directory. Nothing here reads the email.
    provider = GoogleProvider(
        client_id=client_id,
        client_secret=client_secret,
        base_url=base_url,
        required_scopes=["openid", "email"],
    )
    pool = ChildPool(
        root,
        idle_seconds=idle,
        interaction_registration_url=f"{internal_url.rstrip('/')}/_internal/hosted/register",
        interaction_url=base_url,
        # Only when this gateway holds a management key. An operator without one runs exactly
        # the gateway that existed before the trial, and their users meet the OpenRouter
        # approval as they always did.
        interaction_trial_url=(f"{internal_url.rstrip('/')}/_internal/hosted/trial"
                               if trial.enabled() else None),
        max_signins=max_signins,
    )
    # The one durable thing this process writes. It sits at the ROOT of the homes directory and
    # cannot collide with a home: `_SUBJECT_RE` forbids a dot, and this name begins with one.
    ledger = trial.Ledger(Path(os.environ.get("OPYT_TRIAL_LEDGER")
                               or root / ".trial-ledger.json"))
    # Two browser-round-trip stores, in memory beside the interaction nonces and for the same
    # reason: a minted key waiting in a file to be collected is a worse thing to own than a
    # round trip that has to be restarted.
    pending_auth = trial.PendingStore()
    pending_keys = trial.PendingStore()
    resource_metadata = f"{base_url.rstrip('/')}/.well-known/oauth-protected-resource/mcp"

    def unauthorized(detail: str) -> Response:
        # RFC 9728: without this header an MCP client cannot discover which authorization
        # server to use, so a 401 would be a dead end instead of the start of a login.
        return JSONResponse(
            {"error": "unauthorized", "detail": detail}, status_code=401,
            headers={"WWW-Authenticate":
                     f'Bearer resource_metadata="{resource_metadata}"'})

    async def subject_of(request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise PermissionError("missing bearer token")
        access = await provider.verify_token(token)
        if access is None:
            raise PermissionError("token rejected")
        sub = (access.claims or {}).get("sub")
        if not sub:
            raise PermissionError("token carries no subject")
        return str(sub)

    async def proxy(request: Request) -> Response:
        """Authenticate, resolve the child, relay the request and stream the answer back.

        GET, POST and DELETE all forward: Streamable HTTP uses POST for calls, GET for the
        server-to-client stream, and DELETE to end a session.
        """
        try:
            subject = await subject_of(request)
        except PermissionError as e:
            return unauthorized(str(e))

        try:
            child = await pool.acquire(subject)
        except BadSubject as e:
            return JSONResponse({"error": "forbidden", "detail": str(e)}, status_code=403)
        except SpawnFailed as e:
            return JSONResponse({"error": "child_unavailable", "detail": str(e)},
                                status_code=503)

        client: httpx.AsyncClient = request.app.state.http
        body = await request.body()
        upstream = client.build_request(
            request.method,
            f"http://127.0.0.1:{child.port}{MCP_PATH}",
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in _DROP_UPSTREAM},
            params=request.query_params,
            content=body,
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            pool.release(child)
            return JSONResponse({"error": "child_unreachable", "detail": str(e)},
                                status_code=502)

        async def relay():
            # `release` lives here, not in a finally around `send`, because the response is
            # often an SSE stream that outlives the handler. Releasing early would let the
            # reaper end the child while it is still writing to this caller.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                pool.release(child)

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()
                     if k.lower() not in _HOP_BY_HOP},
        )

    async def healthz(request: Request) -> Response:
        # `trial` is here for the LOCAL install's preflight, not for an operator's dashboard: it
        # is what lets a client find out there is no allowance to claim BEFORE it opens a
        # browser tab at this gateway. Without it the only way to learn is to send the user
        # somewhere and let them find a dead end.
        return JSONResponse({"ok": True, "children": pool.snapshot(),
                             "trial": trial.enabled()})

    async def register_hosted_interaction(request: Request) -> Response:
        """Accept a child-minted interaction route without ever touching that child's home."""
        try:
            payload = await request.json()
            nonce = payload["nonce"]
            kind = payload["kind"]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if (not isinstance(nonce, str) or not isinstance(kind, str)
                or not pool.register_interaction(
                    request.headers.get("X-Opyt-Hosted-Interaction-Key", ""), nonce, kind)):
            return JSONResponse({"error": "not_found"}, status_code=404)
        return Response(status_code=204)

    async def login_page(request: Request) -> Response:
        """Consume the URL capability and serve the desktop page it becomes.

        One flow for every site since 2026-09-15: Substack's guided-paste page was deleted the
        day the box was measured actually receiving Substack's sign-in email (the portless
        launch fixed delivery — docs/plans/2026-09-15-substack-adopts-the-x-desktop-flow.md),
        which was the one reason that page existed.
        """
        site = request.path_params["site"]
        if site not in BROWSER_LOGIN_KINDS:
            return _login_error_html_response(None, "That sign-in link is not valid.", 404)
        # BEFORE `consume_interaction`, and that order IS the admission control. Below the pop
        # this check would refuse the user AND spend their link, which is worse than having no
        # cap: a transient full box would cost a capability that clears in minutes, and the user
        # would have to restart onboarding to get another one. Refusing must mutate nothing.
        # Design record: docs/plans/2026-09-08-hosted-signin-admission-control.md, ruling R2.
        live = pool.live_signins()
        if live >= pool.max_signins:
            _log(f"{site} login: refused, {live} sign-in desktops open at the cap of "
                 f"{pool.max_signins} (OPYT_GATEWAY_MAX_SIGNINS); the link was not consumed")
            return _login_error_html_response(
                site, "Too many people are signing in right now.", 503,
                next_step="Wait a minute and reload this page. This link still works.")
        entry = pool.consume_interaction(request.path_params["nonce"], site)
        if entry is None:
            # Three causes share this branch and none reach a child: the link was already
            # used, it outlived its TTL, or the child that minted it was reaped.
            _log(f"{site} login: the link was already used, has expired, or its child is gone")
            return _login_error_html_response(
                site, "This sign-in link was already used or has expired.", 404)
        # The desktop is NOT started here. Its size has to match the box this page will draw
        # it into, and only the page can measure that; a fixed size is 1:1 for one window and
        # shrinks every other (measured, docs/plans/2026-09-09-hosted-desktop-measured-tuning.md).
        # The page reports its box to `login_desktop` below, which starts the desktop.
        session, completion = pool.create_login_session(entry)
        return HTMLResponse(_login_html(site, session, completion),
                            headers={"Cache-Control": "no-store"})

    async def login_desktop(request: Request) -> Response:
        """Start this page's desktop at the size the page measured for it."""
        entry = pool.read_login_session(request.path_params["session"])
        if entry is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        try:
            payload = await request.json()
            width, height = payload["width"], payload["height"]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        size = hosted_browser.clamp_desktop_size(width, height, payload.get("scale", 1))
        try:
            response = await request.app.state.http.post(
                f"http://127.0.0.1:{entry.child.port}/_hosted-login/{entry.nonce}/start",
                json={"width": size.width, "height": size.height, "scale": size.scale},
                headers={"X-Opyt-Hosted-Interaction-Key": entry.child.interaction_key},
                timeout=httpx.Timeout(connect=5.0, read=35.0, write=5.0, pool=5.0),
            )
        except httpx.HTTPError as failure:
            _log(f"login: the child did not answer /start ({type(failure).__name__})")
            return JSONResponse({"error": "not_started"}, status_code=503)
        if response.status_code != 200:
            # The child names the failing step in its own log, and never in this response:
            # the reason describes the server's display stack, not anything a visitor to this
            # public URL is entitled to. Point the operator at the log that holds it.
            _log(f"login: the child refused /start with HTTP {response.status_code}; its "
                 f"reason is in that home's {CHILD_LOG}")
            return JSONResponse({"error": "not_started"}, status_code=503)
        # The framebuffer, not the viewport: the page gives `#screen` this aspect ratio so the
        # box is the desktop's shape before the first frame lands, and at scale 2 the two
        # numbers differ. The ratio is the same either way; sending what noVNC will actually
        # receive keeps the page from having to know that.
        return JSONResponse({"status": "started",
                             "width": size.pixels[0], "height": size.pixels[1],
                             # How long the page may keep re-attaching, from the capability that
                             # decides it. Sending it beats a copy of the TTL in the page, which
                             # would be a second home for a number only this side can know.
                             "expires_in": max(0.0, entry.expires_at - time.monotonic())})

    async def restart_login_browser(request: Request) -> Response:
        """Send a stuck sign-in browser back to its first page.

        The STREAM capability authorizes this, and no new one is minted: whoever holds it can
        already move the mouse and press keys on that desktop through the RFB relay, so being
        able to put the browser back where it started adds no authority. A second token would
        only be a second thing to expire.
        """
        entry = pool.read_login_session(request.path_params["session"])
        if entry is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        pool.hold_login_child(entry.child)
        try:
            response = await request.app.state.http.post(
                f"http://127.0.0.1:{entry.child.port}/_hosted-login/{entry.nonce}/restart",
                headers={"X-Opyt-Hosted-Interaction-Key": entry.child.interaction_key},
                # A restart reaps one Chrome and starts another. The read deadline covers the
                # SIGTERM grace in `_reap_process` plus the settle wait that follows it.
                timeout=httpx.Timeout(connect=5.0, read=25.0, write=5.0, pool=5.0),
            )
        except httpx.HTTPError as failure:
            _log(f"login: the child did not answer /restart ({type(failure).__name__})")
            return JSONResponse({"error": "not_restarted"}, status_code=503)
        finally:
            pool.release_login_child(entry.child)
        if response.status_code != 200:
            _log(f"login: the child refused /restart with HTTP {response.status_code}; its "
                 f"reason is in that home's {CHILD_LOG}")
            return JSONResponse({"error": "not_restarted"}, status_code=503)
        return JSONResponse({"status": "restarted"})

    async def login_stream(websocket: WebSocket) -> None:
        """Relay raw RFB bytes without giving the gateway access to the login desktop."""
        # READ, not consume. A phone drops this socket every time its owner leaves the browser
        # for a 2FA code, and the desktop behind it is still running: single use meant one app
        # switch stranded the user in front of a sign-in they could no longer reach. The
        # one-shot boundary the login URL had is kept by the COMPLETION capability, which is
        # the action, and which `live_signins()` counts.
        entry = pool.read_login_session(websocket.path_params["session"])
        if entry is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        pool.hold_login_child(entry.child)
        try:
            import websockets

            uri = (f"ws://127.0.0.1:{entry.child.port}"
                   f"/_hosted-login/{entry.nonce}/stream")
            async with websockets.connect(
                uri, additional_headers={"X-Opyt-Hosted-Interaction-Key": entry.child.interaction_key},
                open_timeout=5.0, max_size=None, compression=None, proxy=None) as child_socket:
                async def browser_to_child() -> None:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        payload = message.get("bytes")
                        if payload is None:
                            await websocket.close(code=1003)
                            return
                        await child_socket.send(payload)

                async def child_to_phone() -> None:
                    async for message in child_socket:
                        if not isinstance(message, bytes):
                            raise RuntimeError("the child sent text on its RFB relay")
                        await websocket.send_bytes(message)

                outgoing = asyncio.create_task(browser_to_child())
                incoming = asyncio.create_task(child_to_phone())
                done, pending = await asyncio.wait(
                    {outgoing, incoming}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
        except (WebSocketDisconnect, OSError):
            pass
        finally:
            pool.release_login_child(entry.child)
            with contextlib.suppress(Exception):
                await websocket.close()

    async def complete_login(request: Request) -> Response:
        """Relay the page's one-time completion action to the child that owns the profile."""
        entry = pool.consume_login_completion(request.path_params["completion"])
        if entry is None:
            return JSONResponse({"status": "expired"}, status_code=404)
        pool.hold_login_child(entry.child)
        try:
            response = await request.app.state.http.post(
                f"http://127.0.0.1:{entry.child.port}"
                f"/_hosted-login/{entry.nonce}/complete",
                headers={"X-Opyt-Hosted-Interaction-Key": entry.child.interaction_key},
                # ⚠️ This read timeout is sized by the CHILD's work, not by a guess at a healthy
                # latency. Completing waits for Chrome to write the sign-in to disk before it
                # closes the browser (`hosted_browser._LOGIN_SETTLE_SECONDS`, 45 s, because
                # Chromium commits cookies on a ~30 s timer and a kill flushes nothing), then
                # closes the desktop, then makes one authenticated request through a freshly
                # launched Chrome. A timeout under that sum turns a sign-in that WORKED into a
                # 503 the page reads as "could not be checked" — so it moves when that constant
                # moves, and never on its own.
                timeout=httpx.Timeout(connect=5.0, read=150.0, write=5.0, pool=5.0),
            )
        except httpx.HTTPError:
            response = None
        finally:
            pool.release_login_child(entry.child)
        if response is None or response.status_code != 200:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse(response.json())

    async def openrouter_callback(request: Request) -> Response:
        """Receive OpenRouter's authorization code and relay it straight to its child.

        The callback's nonce is a one-time route capability. The child retained the matching
        PKCE verifier, exchanges the code, and writes the resulting key inside its own home.
        """
        code = request.query_params.get("code")
        if not code or len(code) > 4096:
            return HTMLResponse(_openrouter_result_html(False), status_code=400,
                                headers={"Cache-Control": "no-store",
                                         "Referrer-Policy": "no-referrer"})
        entry = pool.consume_interaction(request.path_params["nonce"], "openrouter")
        if entry is None:
            return HTMLResponse(_openrouter_result_html(False), status_code=404,
                                headers={"Cache-Control": "no-store",
                                         "Referrer-Policy": "no-referrer"})
        try:
            response = await request.app.state.http.post(
                f"http://127.0.0.1:{entry.child.port}"
                f"/_hosted-openrouter/callback/{entry.nonce}",
                json={"code": code},
                headers={"X-Opyt-Hosted-Interaction-Key": entry.child.interaction_key},
                timeout=httpx.Timeout(connect=5.0, read=35.0, write=5.0, pool=5.0),
            )
        except httpx.HTTPError:
            response = None
        stored = response is not None and response.status_code == 200
        return HTMLResponse(_openrouter_result_html(stored), status_code=200 if stored else 400,
                            headers={"Cache-Control": "no-store",
                                     "Referrer-Policy": "no-referrer"})

    # ── The starter allowance ──────────────────────────────────────────────────────────────
    # Two doors to one mint, because the two homes answer "who is asking" in completely
    # different ways. A hosted child was spawned BY this gateway for an already-authenticated
    # subject, so it only has to prove it is that child. A local install is a stranger, so it
    # brings its user through a Google sign-in first. Both land in `trial.mint`, and the ledger
    # they share is what keeps the allowance one per person across the two.

    async def hosted_trial(request: Request) -> Response:
        """Mint for the child that proves it holds an interaction key.

        The SUBJECT IS READ FROM THE POOL, never from the request. A child that could name its
        own subject could claim one allowance per name it invented, and it is the only party
        with a reason to try.
        """
        if not trial.enabled():
            return JSONResponse({"error": "not_configured"}, status_code=404)
        child = pool.child_for_key(
            request.headers.get("X-Opyt-Hosted-Interaction-Key", ""))
        if child is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        try:
            minted = await trial.mint(request.app.state.http, child.subject, ledger)
        except trial.TrialUnavailable as e:
            # LOGGED HERE, and only here. The operator needs the provider's own sentence — an
            # unreadable refusal is an unfixable one — while the child gets only the short word,
            # so nothing an upstream says can leak into what a user is shown.
            if getattr(e, "detail", ""):
                _log(f"trial mint refused for {child.subject}: {e} - {e.detail}")
            # 409, not 500: every one of these is a settled answer about this subject, and the
            # child must read it as "take the other path" rather than as "try again".
            return JSONResponse({"error": str(e)}, status_code=409)
        return JSONResponse(minted)

    def _trial_bounce(callback: str, params: dict) -> Response:
        return RedirectResponse(f"{callback}?{urlencode(params)}", status_code=302,
                                headers={"Cache-Control": "no-store",
                                         "Referrer-Policy": "no-referrer"})

    def _trial_dead_end(detail: str) -> Response:
        """When there is no callback to bounce to, the browser is the only thing left holding
        the user, so it gets a real page rather than a status code."""
        return HTMLResponse(
            web_panel.render("Opyt", ok=False,
                             headline="That link cannot be used.", detail=detail),
            status_code=400, headers={"Cache-Control": "no-store",
                                      "Referrer-Policy": "no-referrer"})

    async def trial_start(request: Request) -> Response:
        """Begin the local flow: validate the loopback callback, then hand off to Google."""
        callback = trial.loopback_callback(request.query_params.get("callback_url", ""))
        challenge = request.query_params.get("code_challenge", "")
        if callback is None or not challenge:
            return _trial_dead_end("Start setup again from your Claude conversation.")
        if request.query_params.get("code_challenge_method") != "S256":
            return _trial_bounce(callback, {"error": "bad_request"})
        if not trial.enabled():
            return _trial_bounce(callback, {"error": "not_configured"})
        state = pending_auth.put({"callback": callback, "challenge": challenge})
        # `openid` alone: `sub` is the only claim this flow uses, and asking for an email we
        # would never read is a consent screen that overstates what OPYT learns.
        return RedirectResponse(f"{trial.GOOGLE_AUTH}?" + urlencode({
            "client_id": client_id,
            "redirect_uri": f"{base_url.rstrip('/')}/trial/google",
            "response_type": "code", "scope": "openid", "state": state,
            "prompt": "select_account"}), status_code=302)

    async def trial_google(request: Request) -> Response:
        """Google comes back here. Resolve the subject, mint, and hand the local process a
        one-time code — never the key, which would be putting a live credential in a URL."""
        held = pending_auth.take(request.query_params.get("state", ""))
        if held is None:
            return _trial_dead_end("That sign-in link has already been used or has expired.")
        callback = held["callback"]
        if request.query_params.get("error") or not request.query_params.get("code"):
            return _trial_bounce(callback, {"error": "declined"})
        try:
            token = await request.app.state.http.post(
                trial.GOOGLE_TOKEN,
                data={"code": request.query_params["code"],
                      "client_id": client_id, "client_secret": client_secret,
                      "redirect_uri": f"{base_url.rstrip('/')}/trial/google",
                      "grant_type": "authorization_code"},
                timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
        except httpx.HTTPError:
            return _trial_bounce(callback, {"error": "upstream"})
        subject = (trial.subject_from_id_token((token.json() or {}).get("id_token", ""))
                   if token.status_code < 300 else None)
        if not subject:
            return _trial_bounce(callback, {"error": "upstream"})
        try:
            minted = await trial.mint(request.app.state.http, subject, ledger)
        except trial.TrialUnavailable as e:
            return _trial_bounce(callback, {"error": str(e)})
        code = pending_keys.put({"payload": minted, "challenge": held["challenge"]})
        return _trial_bounce(callback, {"code": code})

    async def trial_exchange(request: Request) -> Response:
        """Hand the key to the process that started the flow, and to nothing else.

        ONE ATTEMPT, pass or fail — ordinary authorization-code semantics. The code is consumed
        by being read, so a verifier that does not match burns it rather than leaving a second
        try for whoever holds the code.
        """
        try:
            body = await request.json()
            code, verifier = body["code"], body["code_verifier"]
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        held = pending_keys.take(code if isinstance(code, str) else "")
        if held is None or not isinstance(verifier, str):
            return JSONResponse({"error": "not_found"}, status_code=404)
        if not trial.verify_pkce(verifier, held["challenge"]):
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse(held["payload"], headers={"Cache-Control": "no-store"})

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        # No read timeout: an MCP session's GET stream stays open for the whole session, and a
        # read timeout would sever it. The connect timeout stays, because that one is a real
        # network call that can hang.
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=5.0))
        app.state.pool = pool
        pool.start_reaper()
        try:
            yield
        finally:
            await app.state.http.aclose()
            await pool.shutdown()

    routes = provider.get_routes(mcp_path=MCP_PATH)
    routes += [
        Mount("/static", app=StaticFiles(directory=_STATIC_ROOT), name="static"),
        Route(MCP_PATH, proxy, methods=["GET", "POST", "DELETE"]),
        Route("/_internal/hosted/register", register_hosted_interaction, methods=["POST"]),
        Route("/_internal/hosted/trial", hosted_trial, methods=["POST"]),
        Route("/trial/start", trial_start, methods=["GET"]),
        Route("/trial/google", trial_google, methods=["GET"]),
        Route("/trial/exchange", trial_exchange, methods=["POST"]),
        # Declared BEFORE the generic sign-in routes: OpenRouter is a callback, not a
        # desktop, and `/login/{site}/{nonce}` would otherwise swallow its path.
        Route("/login/openrouter/{nonce}", openrouter_callback, methods=["GET"]),
        Route("/login/complete/{completion}", complete_login, methods=["POST"]),
        # Both sit before the catch-all `/login/{site}/{nonce}`, which would otherwise match
        # "desktop" and "restart" as site names.
        Route("/login/desktop/{session}", login_desktop, methods=["POST"]),
        Route("/login/restart/{session}", restart_login_browser, methods=["POST"]),
        Route("/login/{site}/{nonce}", login_page, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        WebSocketRoute("/login/session/{session}", login_stream),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


# What a sign-in page says about the site it is connecting. The desktop, the relay and the
# completion action are identical for every site; only these two strings are not.
_LOGIN_SITES = {
    "x": {
        "name": "X",
        # No note. Nothing about signing in to X needs saying that the page does not already
        # say -- it is a browser showing x.com. `.note:empty` removes the element entirely.
        "note": "",
    },
    "substack": {
        "name": "Substack",
        # Empty since 2026-09-16, by David's call after his first successful desktop sign-in.
        # It carried a warning not to tap the email's link, and the sentence had two problems:
        # the consequence it stated ("the code stops working") was an INFERENCE from the
        # converse measurement and was never measured in that direction, and the page it sat
        # under already shows Substack's own code boxes, which say what to do. Do not restore
        # it without measuring the claim first.
        "note": "",
    },
}


def _login_html(site: str, session: str, completion: str) -> str:
    """The consumed URL becomes binary-stream and completion capabilities for this one page."""
    copy = _LOGIN_SITES[site]
    return (_LOGIN_PAGE
            .replace("__CSS__", _LOGIN_CSS)
            .replace("__SITE__", copy["name"])
            .replace("__NOTE__", copy["note"])
            .replace("__STREAM__", json.dumps(f"/login/session/{session}"))
            .replace("__DESKTOP__", json.dumps(f"/login/desktop/{session}"))
            .replace("__RESTART__", json.dumps(f"/login/restart/{session}"))
            .replace("__COMPLETE__", json.dumps(f"/login/complete/{completion}")))


# The shell of the sign-in page: the brand mark, the type scale, the palette and the
# completion panel. Split from the page so the chrome reads as one unit.
_LOGIN_CSS = """:root { color-scheme: light; --bg:#fff; --surface:#f4f4f4; --text:#141414; --muted:#5c5c5c;
  --faint:#8f8f8f; --accent:#dd3418; --line:#e3e3e3; }
* { box-sizing: border-box; }
body { min-height:100vh; margin:0; background:var(--bg); color:var(--text); font:14px/1.6
  ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
/* One flex column the height of the viewport; .stage takes whatever the three short rows
   around it do not, and #screen sits centred inside it at the desktop's own shape. noVNC
   scales the whole display into #screen, so its height IS the scale factor: at
   `min(900px, 72vh)` a 900px desktop rendered at about 0.63 and was never close to 1:1.
   Measured 2026-09-09, docs/plans/2026-09-09-hosted-desktop-measured-tuning.md. */
.shell { display:flex; flex-direction:column; gap:.5rem; height:100vh;
  height:var(--visual-height, 100dvh);
  width:min(1600px, 100%); margin:0 auto; padding:1.75rem 1rem 1.25rem; }
/* One row, one type size, centred. The brand and the heading were 14px and 16px sitting on a
   shared baseline next to a 22px tile, which read as three mismatched things rather than one
   line. Colour separates them now, so no separator glyph and no size change is needed. */
.top { display:flex; align-items:center; gap:.9rem; }
.brand { display:flex; align-items:center; gap:.45rem; font-size:.95rem; font-weight:600;
  color:var(--muted); }
.tile { width:20px; height:20px; border-radius:5px; }
/* A rule, because a gap alone did not read as one: in monospace, "Opyt" and the heading at the
   same size ran together as a single phrase. The divider says which is the app and which is
   the task. */
h1 { margin:0; padding-left:.9rem; border-left:1px solid var(--line); font-size:.95rem;
  font-weight:600; }
/* Prose stops at a readable measure. Full-bleed monospace across 1280px is a wall. */
#status { margin:0; max-width:76ch; color:var(--muted); font-size:.82rem; }
.stage { position:relative; flex:1; min-height:0; display:grid; place-items:center; }
/* While the on-screen keyboard is up, the stage stops being a frame around the whole desktop
   and becomes a window onto part of it, at the size the desktop already had. `align-items` has
   to leave centring for `start`: a centred item that overflows spills equally above and below,
   and the half above the top edge is not reachable by scrollTop. `/static/desktop.js` sets the
   class, freezes the height, and scrolls this to wherever the user last tapped. */
.stage.typing { align-items:start; overflow:hidden; }
/* What a phone's on-screen keyboard actually types into. It has to be focusable, so it can be
   neither display:none nor visibility:hidden, and it has to cost the layout nothing. It sits
   inside .stage so that scrolling it into view -- which iOS does on focus, whatever we ask --
   scrolls to the desktop rather than away from it. */
#keys { position:absolute; top:0; left:0; width:1px; height:1px; padding:0; border:0;
  opacity:0; pointer-events:none; }
/* noVNC contain-fits the framebuffer (autoscale() in core/display.js picks the smaller of the
   two axis ratios), so any element that is not the desktop's own shape gets black bands. The
   real ratio is set from the canvas on connect -- the display size lives in another process,
   and a copy of it here would be a second source of truth that rots silently. This 16/9 is only
   what the empty box looks like for the ~150 ms before the stream arrives, and the width cap on
   `.shell` above has to be chosen alongside the display size, not independently of it. */
#screen { width:100%; aspect-ratio:16/9; max-height:100%; overflow:hidden;
  border:1px solid var(--line); border-radius:10px;
  background:var(--surface); }.live { border-color:var(--accent); box-shadow:0 0 0 3px
  rgba(221,52,24,.13); }
.actions { display:flex; align-items:center; justify-content:space-between; gap:1.5rem; }
/* margin-left:auto, not justify-content alone: with X's note empty the buttons are the row's
   only child, and `space-between` puts a lone item at the START. */
.buttons { display:flex; gap:.6rem; margin-left:auto; }
/* The two secondary controls share one outline; only the action that ENDS the sign-in is
   filled, so a mis-tap on the row costs a tap and never the sign-in. */
#keyboard, #restart { min-height:2.6rem; border:1px solid var(--line); border-radius:7px;
  padding:0 1.1rem; background:var(--bg); color:var(--text); font:600 .82rem inherit;
  white-space:nowrap; cursor:pointer; }
#restart:disabled { color:var(--faint); cursor:default; }
/* A canvas cannot raise a phone's keyboard, so there is a button that can. Pointer-coarse and
   not a width query: what decides whether this is needed is having no physical keys, and a
   narrow window on a laptop has plenty. */
#keyboard { display:none; }
#keyboard[aria-pressed=true] { border-color:var(--accent); color:var(--accent); }
@media (pointer: coarse) { #keyboard { display:block; } }
.note { margin:0; max-width:76ch; color:var(--faint); font-size:.78rem; }
.note:empty { display:none; }
/* The status line's own spinner, for the states where the page is WAITING on the server rather
   than on the user. The completion check is the one that needed it: the child holds the browser
   open until Chrome has committed the session to disk (up to 45 s) before it may close it and
   look, and a sentence that never moves reads as a hung tab — the one moment a user must not
   give up on a sign-in that is about to succeed. Sized in `em` so it tracks the status text,
   and drawn from a border so it needs no asset and inherits the text colour. */
.spin { display:none; width:.8em; height:.8em; margin-right:.5em; vertical-align:-.08em;
  border:2px solid currentColor; border-top-color:transparent; border-radius:50%;
  animation:spin .7s linear infinite; }
#status.busy .spin { display:inline-block; }
@keyframes spin { to { transform:rotate(360deg); } }
/* Slowed rather than stopped: the progress signal is the point, and a motion-sensitive user
   needs it as much as anyone during a 45-second wait. */
@media (prefers-reduced-motion: reduce) { .spin { animation-duration:2.4s; } }
#complete { min-height:2.6rem; border:0; border-radius:7px; padding:0 1.1rem;
  background:var(--text);
  color:var(--bg); font:600 .82rem inherit; white-space:nowrap; cursor:pointer; }
#complete:hover { background:var(--accent); } #complete:disabled { background:var(--line);
  color:var(--faint); cursor:default; }
#done { display:none; place-items:center; width:100%; height:100%; border:1px solid var(--line);
  border-radius:10px; background:var(--surface); text-align:center; padding:2rem 1rem; }
#done.show { display:grid; }
#done .mark { display:grid; width:3.4rem; height:3.4rem; margin:0 auto 1.1rem; place-items:center;
  border-radius:50%; background:var(--accent); color:#fff; font-size:1.7rem; line-height:1; }
#done.failed .mark { background:var(--faint); }
#done h2 { margin:0 0 .5rem; font-size:1.15rem; letter-spacing:-.02em; }
#done p { margin:0; color:var(--muted); }
/* A phone. The column stays exactly the height of what the user can see -- `--visual-height`
   above tracks the on-screen keyboard, which no CSS length reports -- because the desktop is
   the page here, and a scrolling column would push half of it under the keys. The rows around
   it are one line each, so the stage still gets most of the screen. */
@media (max-width:560px) { .shell { padding:.9rem .7rem 1rem; gap:.45rem; }
  .top { gap:.6rem; } h1 { padding-left:.6rem; }
  #status { font-size:.78rem; }
  .actions { flex-direction:column; align-items:stretch; gap:.5rem; }
  /* Two rows, not three: the secondaries share the first and the primary takes the second on
     its own. `flex-basis:100%` is what wraps it, so it needs no `order` and no second row in
     the markup. */
  .buttons { margin-left:0; flex-wrap:wrap; }
  /* 44 CSS px is the smallest target a thumb hits reliably, and the desktop below is drawn
     from whatever these rows leave, so they are no taller than that either. */
  #keyboard, #restart, #complete { min-height:2.75rem; }
  #keyboard, #restart { flex:1 1 0; padding:0 .5rem; }
  #complete { flex:1 0 100%; } }
"""


# The noVNC source is vendored at `gateway/static/novnc/` (version and license travel with it),
# so this login does not acquire a third-party script at the moment a user needs to authenticate.
_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=theme-color content="#ffffff"><title>Connect __SITE__ · Opyt</title><style>
__CSS__</style></head><body><main class=shell>
<div class=top><div class=brand><svg class=tile viewBox="0 0 64 64" aria-hidden="true"><rect
width="64" height="64" fill="#141414"/><path d="M18 16 L36 32 L18 48" stroke="#ffffff"
stroke-width="7" fill="none"/><rect x="40" y="41" width="11" height="7" fill="#dd3418"/></svg>Opyt
</div><h1>Connect your __SITE__ account</h1></div>
<p id=status><span class=spin aria-hidden="true"></span><span id=status-text>Opening a private
sign-in desktop…</span></p>
<noscript><p id=status>This page needs JavaScript to show the sign-in desktop.</p></noscript>
<div class=stage>
<textarea id=keys autocomplete=off autocorrect=off autocapitalize=off spellcheck=false
  aria-label="Type into the __SITE__ sign-in desktop" tabindex=-1></textarea>
<div id=screen aria-label="__SITE__ sign-in desktop"></div>
<div id=done role=status aria-live=polite><div><p class=mark aria-hidden="true">&#10003;</p>
<h2 id=done-title></h2><p id=done-detail></p></div></div>
</div>
<div class=actions id=actions><p class=note>__NOTE__</p>
<div class=buttons><button id=keyboard type=button aria-pressed=false
  data-show="Keyboard" data-hide="Hide keyboard">Keyboard</button>
<button id=restart type=button>Start over</button>
<button id=complete type=button>Done signing in</button></div></div>
</main><script type=module>
import {attachDesktop} from '/static/desktop.js';
const status = document.getElementById('status');
const statusText = document.getElementById('status-text');
const screen = document.getElementById('screen');
const button = document.getElementById('complete');
const done = document.getElementById('done');
let finished = false;

// One place owns the status line, so the spinner and the sentence cannot disagree: every state
// that leaves the user WAITING on the server spins, and every state that asks them to do
// something does not. Writing `status.textContent` directly would also delete the spinner,
// since it is a child of that element.
function setStatus(text, busy) {
  statusText.textContent = text;
  status.classList.toggle('busy', Boolean(busy));
}
// Starting the desktop is itself a wait — an Xvfb, an x11vnc and a Chrome, measured at 1.9-9.9 s
// on the box — so the page spins from the moment its script runs. The served markup carries no
// `busy` class on purpose: with JavaScript off nothing here will ever finish, and a spinner
// that turns forever would promise otherwise.
setStatus('Opening a private sign-in desktop…', true);

// The desktop is created at the size of the box it will be drawn into, which is why this page
// starts it rather than the route that served the page. noVNC scales the whole display into
// #screen and #screen is given the display's aspect ratio, so both terms of the fit are equal
// and the scale is exactly elementHeight / displayHeight. Matching them makes it 1.0 on every
// window; any fixed size is 1:1 for one window and shrinks the rest. Measured 2026-09-09:
// a fixed 1600x900 gave 0.98 on a 1695x1060 viewport and 0.71 on a 1440x790 laptop.
const stage = document.querySelector('.stage').getBoundingClientRect();
let started;
try {
  const response = await fetch(__DESKTOP__, {
    method: 'POST', credentials: 'same-origin',
    headers: {'Content-Type': 'application/json'},
    // The box in CSS pixels, and how many device pixels each of those is. A phone drawing a
    // 480-pixel-wide desktop into a 390-pixel box at devicePixelRatio 3 puts one remote pixel
    // under 2.4 physical ones, and text survives that only as a blur; asking for the desktop
    // at 2 makes it nearly native. The server decides what it can afford.
    body: JSON.stringify({width: Math.round(stage.width), height: Math.round(stage.height),
                          scale: window.devicePixelRatio >= 2 ? 2 : 1}),
  });
  if (!response.ok) { throw new Error(String(response.status)); }
  started = await response.json();
} catch (error) {
  finish(false, 'The sign-in desktop could not start',
         'Go back to Claude and start a new sign-in.');
  throw error;
}
// The size the server actually used, after its own clamp. Setting it here rather than reading
// it back off noVNC's canvas means the box is the right shape before the first frame lands.
screen.style.aspectRatio = started.width + ' / ' + started.height;

// 'live' is an instruction, not a status. The stream is live within ~150 ms but Chrome needs
// 2-10 s (measured on the box, 2026-09-08) to paint the site, so it fires while the screen
// below is still a blank white rectangle. Saying "connected" there described the RFB stream and
// was read as "my account is connected" -- the opposite of the truth at the one moment the
// visitor has to act. Telling them what to do covers the same wait without the false claim.
//
// The button is named here exactly as it is labelled, and that label states the ORDER on
// purpose. The button does not start a sign-in: it ends the desktop and checks the session, so
// a visitor who presses it first fails their own connect. A label like "Sign-in" would invite
// exactly that press.
//
// A phone gets a different sentence, because it has a step a laptop does not: a canvas cannot
// raise an on-screen keyboard, so a tap on the desktop is what opens it. Tapping does raise it
// on its own (`/static/desktop.js`), and this says so anyway -- a user whose first tap lands on
// a button rather than a field has to know that typing is available at all.
const SIGN_IN_STEPS = window.matchMedia('(pointer: coarse)').matches
  ? "Sign into __SITE__: tap the screen where you want to type. Then tap 'Done signing in'."
  : "Sign into __SITE__. Then click the 'Done signing in' button at the bottom.";
function onStatus(state) {
  if (finished) { return; }
  screen.classList.toggle('live', state === 'live');
  if (state === 'live') {
    setStatus(SIGN_IN_STEPS);
  } else if (state === 'lost') {
    // Not a failure, and the button beneath it still works: completing is an HTTP call to the
    // child that owns the profile, and the desktop this lost sight of is still running there.
    // Leaving the browser for a code is the ordinary way to sign in on a phone, and it costs
    // the socket every time. It spins because the page is retrying by itself.
    setStatus('Reconnecting to the sign-in desktop…', true);
  } else {
    setStatus('The sign-in desktop closed. Return to Opyt and try again.');
    button.disabled = true;
  }
}

const desktop = attachDesktop({
  screen, stage: document.querySelector('.stage'), keys: document.getElementById('keys'),
  button: document.getElementById('keyboard'),
  streamPath: __STREAM__, expiresIn: started.expires_in, onStatus,
});
// Every outcome ends the same way: the desktop is gone and the person has to go back to the
// app that sent them. Saying so in the status line left that instruction as one small sentence
// above the largest thing on the page — an empty grey rectangle where the browser used to be.
// The panel takes that space, because after completion the screen has nothing left to show.
function finish(ok, title, detail) {
  finished = true;
  screen.style.display = 'none';
  document.getElementById('actions').style.display = 'none';
  setStatus('');
  done.classList.toggle('failed', !ok);
  done.querySelector('.mark').textContent = ok ? '\u2713' : '!';
  document.getElementById('done-title').textContent = title;
  document.getElementById('done-detail').textContent = detail;
  done.classList.add('show');
}
// The way out of a page with no way out. A sign-in page is mostly buttons that open other
// sign-ins, and X's "Continue with Google" opens a Chrome popup window with no back button,
// no tab strip and no title bar (measured 2026-09-09; the desktop runs no window manager).
// Nothing inside that window can dismiss it, so the control lives out here instead.
const restart = document.getElementById('restart');
restart.onclick = async () => {
  restart.disabled = true;
  setStatus('Reopening the __SITE__ sign-in page\u2026', true);
  try {
    const response = await fetch(__RESTART__, {method:'POST', credentials:'same-origin'});
    if (!response.ok) { throw new Error(String(response.status)); }
  } catch (_) {
    // The desktop kept its display and its stream either way, so this is not a finish(): what
    // the user has lost is the browser on it, and only a new sign-in brings one back.
    setStatus('The sign-in page could not be reopened. Go back to Claude and start \
a new sign-in.');
    return;
  }
  setStatus(SIGN_IN_STEPS);
  restart.disabled = false;
};
button.onclick = async () => {
  // Before the request, not after: completing is what closes the desktop, so the RFB drop that
  // follows is this click succeeding. Left running, the reconnect above would spend the rest of
  // the window trying to re-attach to a desktop this click is in the middle of ending.
  finished = true;
  desktop.stop();
  button.disabled = true;
  // Named as a wait, because it is one: the child holds the browser open until Chrome has
  // written the session to disk (up to 45 s) before it may close it and check. A bare
  // "Checking…" on a phone for that long reads as a hung page, and the one thing a user must
  // not do here is give up on a sign-in that is about to succeed. This is the wait the
  // spinner was added for (2026-09-16), because the sentence alone was not enough.
  setStatus('Checking your __SITE__ connection — this can take up to a minute…', true);
  try {
    const response = await fetch(__COMPLETE__, {method:'POST', credentials:'same-origin'});
    const result = await response.json();
    if (result.status === 'connected') {
      finish(true, '__SITE__ is connected', 'Go back to your Claude conversation to carry on.');
    } else if (result.status === 'not_connected') {
      finish(false, '__SITE__ is not connected',
             'Go back to Claude and start a new sign-in.');
    } else {
      finish(false, 'This sign-in expired', 'Go back to Claude and start a new sign-in.');
    }
  } catch (_) {
    finish(false, 'The connection could not be checked',
           'Go back to Claude and start a new sign-in.');
  }
};
</script></body></html>"""


def _login_error_html_response(site: str | None, message: str, status_code: int, *,
                               next_step: str | None = None) -> HTMLResponse:
    """What a person sees when a sign-in link cannot be opened, in place of a JSON error.

    This URL is clicked from a chat, so its failures land in a browser. The text says what to
    do next and never why the server refused. `site` is None when the path segment itself was
    not a site, which is the one case where there is no name to print.

    `next_step` exists because the default sentence is wrong for exactly one caller. Every other
    failure here has already spent the link, so "ask to connect X again" is the only way
    forward. A capacity refusal spends nothing, and telling that visitor to start over would
    throw away a capability that still works.
    """
    name = _LOGIN_SITES[site]["name"] if site in _LOGIN_SITES else None
    title = f"Connect {name} to Opyt" if name else "Connect a source to Opyt"
    again = f"ask to connect {name} again" if name else "start the sign-in again"
    return HTMLResponse(
        web_panel.render(title, ok=False, headline=message,
                         detail=next_step or f"Go back to Claude and {again}."),
        status_code=status_code, headers={"Cache-Control": "no-store"})


def _openrouter_result_html(stored: bool) -> str:
    """A callback confirmation that never repeats the authorization code into the page."""
    if stored:
        return web_panel.render("Connect OpenRouter to Opyt", ok=True,
                                headline="OpenRouter is connected",
                                detail="Go back to your Claude conversation to carry on.")
    return web_panel.render("Connect OpenRouter to Opyt", ok=False,
                            headline="OpenRouter could not be connected",
                            detail="Go back to Claude and ask to connect OpenRouter again.")
