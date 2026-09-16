# Running the gateway

The public HTTPS endpoint claude.ai points at. It authenticates with Google, resolves the caller
to a `sub` claim, and proxies to that user's own `opyt-mcp` child on loopback. Design record:
`docs/plans/2026-09-02-hosted-opyt-remote-connector.md`.

## Google Cloud setup — do this first

Create an **OAuth 2.0 Client ID** of type *Web application*, and set one **authorized redirect
URI**:

```
<OPYT_GATEWAY_BASE_URL>/auth/callback
```

That path is `GoogleProvider`'s, not ours, and it is the single value that silently breaks the
whole login if it is wrong — Google rejects the redirect and the user sees a Google error page
rather than anything from Opyt. Register the loopback URI too
(`http://127.0.0.1:8080/auth/callback`), which lets the whole login be tested with no domain and
no TLS; Google exempts loopback from its https rule. Use `127.0.0.1`, not `localhost`: this Mac
resolves that name to `::1` first and the gateway binds IPv4.

**Leave the App logo field empty.** Uploading one forces app verification unless the publishing
status stays "Testing", which caps the app at 100 hand-listed test users who each see a warning
screen. `openid` and `email` are non-sensitive, so with no logo the app publishes to production
with no review and no cap.

**Consent-screen fields:** app name `Opyt`, home page `https://useopyt.com`, authorized domain
`useopyt.com` (bare, no scheme — it covers `mcp.useopyt.com`; loopback needs no entry). The
privacy-policy field stays blank until `website/privacy.html` exists (queued in the plan).

Scopes are requested by the code and normalized to Google's full URIs, so the metadata advertises
`openid` and `https://www.googleapis.com/auth/userinfo.email`. A client that registers with the
shorthand `email` instead is rejected at `/authorize`; clients that read `scopes_supported` from
the metadata, as claude.ai does, get this right on their own.

Google is **not** the authorization server claude.ai talks to. claude.ai requires dynamic client
registration and Google does not offer it, so the gateway is the authorization server and Google
is the upstream login behind it. Swapping Google out is invisible to claude.ai.

## Environment

| variable | required | meaning |
|---|---|---|
| `OPYT_GATEWAY_BASE_URL` | yes | the **public https** URL, e.g. `https://mcp.useopyt.com`. It is what the OAuth metadata advertises and what Google redirects to. |
| `OPYT_GATEWAY_GOOGLE_CLIENT_ID` | yes | from the console |
| `OPYT_GATEWAY_GOOGLE_CLIENT_SECRET` | yes | from the console |
| `OPYT_HOMES_ROOT` | no | where per-user homes live. Default `~/.opyt-homes`; on a server, `/srv/homes`. |
| `OPYT_GATEWAY_IDLE_SECONDS` | no | reap a child after this long with no traffic. Default 900. |
| `OPYT_GATEWAY_HOST` / `OPYT_GATEWAY_PORT` | no | bind address. Default `127.0.0.1:8080` — terminate TLS in front. |
| `OPYT_HOSTED_CHROME` | no | absolute Chrome/Chromium executable for the hosted browser boundary; omit when `google-chrome`, `google-chrome-stable`, `chromium`, or `chromium-browser` is on `PATH`. |
| `OPYT_WORKER_DB` | yes, when a worker runs | the rail worker's jobs database, e.g. `/var/lib/opyt-worker/rail_jobs.db`. **Inherited by every child** and must match `opyt-worker.service` exactly. |

All four `OPYT_GATEWAY_*` values plus `OPYT_HOMES_ROOT` are stripped from every child's
environment. `X_WEB_BEARER` is stripped too: hosted X uses Chrome's per-home profile, never an
operator-provided bearer (that profile now also holds the hosted Substack session). A per-user process has no use for either value.

`OPYT_WORKER_DB` is the one variable that deliberately goes the other way, because a child is
where a product action asks for unattended work. Alongside it the gateway sets
`OPYT_WORKER_HOME_ID` to the subject it has already validated, which is how a child names its
own home in the jobs table and the reason it cannot name anyone else's. No tool argument sets
either one. A child with a home id and no jobs database refuses to queue rather than writing
into `$OPYT_HOME`, where the worker would never look.

## Run

```bash
python -m gateway
```

**One process. Never `--workers`.** Each worker would keep its own routing table and spawn its
own child per user, putting two `opyt-mcp` processes on one home. Every route here is I/O, so a
second worker buys nothing. Scale by sharding users across boxes.

`/healthz` lists the live children (subject, pid, port, in-flight count, idle seconds).

## Adding the connector

In claude.ai, add a custom connector pointing at `<OPYT_GATEWAY_BASE_URL>/mcp`. The Google login
happens on first use.

## What runs where

| process | owns |
|---|---|
| gateway | OAuth, subject→child routing, reaping, and short-lived hosted interaction routing |
| child (`opyt-mcp --http <port>`) | one `$OPYT_HOME`: its store, settings, onboarding state, Chrome profile, and temporary X/VNC desktop |

The gateway never reads a home, never owns a browser profile or X credential state, and holds no
credential of theirs. The child authenticates nobody — it binds loopback, and the gateway is the
only thing that can reach it. Hosted X and OpenRouter approval links are one-use and short-lived;
access logging stays off so their capabilities never reach logs.

Hosted X requires Chrome/Chromium plus `Xvfb` and `x11vnc` on the server. A short-lived login
gets its own X display, headed Chrome without a debugging port, and a loopback-only VNC listener;
the gateway relays that RFB stream to the vendored noVNC client. The child closes and reaps all
three processes on completion, expiry, or shutdown.

**The rails do not run in a child.** A hosted child is disposable and gets SIGTERMed when idle,
so unattended ingest belongs to `opyt-worker`, a separate systemd service that shares the
checkout and reads the jobs database. A child's only part in it is queuing a durable job for its
own home; the worker owns every rail launch and its result.

Product actions queue durable work: onboarding consent queues the bookmark and (when the roster
is non-empty) Oracle rails and its curation phase queues the curation rail, sharing queues the push
rail, and a sitting read queues the sitting scheduler. Three more are chained — a sitting read that
emits standing queries makes Frontier stage 2 due, a stage-2 pass that stages candidates makes
stage 3 due, and a curation pass that pulled makes the candidate probe due. The worker owns every
launch; a home with no due work is intentionally dormant. The MCP server, in either mode, starts
nothing.
