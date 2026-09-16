#!/usr/bin/env python3
"""
OPYT Knowledge Base MCP Server

The registration point for every tool module, and nothing else — this module defines no
tools of its own. Retrieval is served by `atoms_tools` (`search`, `open`, `aggregate`) over
the atom KB (atoms + chunks + chunks_fts in ~/.opyt/opyt.db).

`main()` starts NO background work. The resident `opyt-worker` process is the only thing that
launches a rail; a tool that has just written a consent, grant, or claim makes one due through
`pipeline/kb/rail_jobs.request_now`. Design record:
docs/plans/2026-09-06-persistent-rail-worker-migration.md.

Usage:
  python mcp_server/server.py                 stdio — the local install
  python mcp_server/server.py --http <port>   HTTP on 127.0.0.1:<port> — a hosted child
"""

import sys
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import Icon

# Make the repo root importable. Claude Code launches this as `python mcp_server/server.py`,
# which puts mcp_server/ on sys.path but NOT the repo root — so `import opyt_core` / `import
# pipeline` / `from mcp_server.x import` inside the registration blocks below would fail.
# Since this module defines no tools of its own, a failed insert yields a server with an EMPTY
# tool surface. Inserting the repo root here fixes registration regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _frontier_session_notice() -> str:
    """The frontier's state at session start — the WHOLE of the server instructions, or "".

    Built once per session (one process per stdio MCP session), so this is the only channel the
    frontier can push through without the user calling something first. A count and a pointer,
    never the list — stays one sentence, silent (returns "") when there's nothing to report.
    Wrapped: a DB hiccup at startup must never stop the server serving (CLAUDE.md fail-safe).
    """
    try:
        from mcp_server.frontier_tools import notice
        n = notice()
    except Exception:
        return ""
    if not n:
        return ""
    return (f"\n\nFRONTIER STATE AT SESSION START: {n['unshown']} staged artifact"
            f"{'s' if n['unshown'] != 1 else ''} the user has not been shown yet (newest: "
            f"{n['top']['title'][:90]!r}). Call `frontier` to see the ranked queue — but only "
            f"when it is relevant to what the user is doing, or when they ask what is new. Do "
            f"not open a session by reciting this.")


# ── The standing citation rule ──────────────────────────────────────────────────
# RULED 2026-09-16, after the tool-description channel was measured failing FOUR times. The link
# is on every hit three ways over (`cite`, `source_url`, and for a while inside `description`),
# and a rule in `search`'s own description told the host to show it — and four hosted phone
# answers (Sonnet 5 Medium) still came back as linkless digests. Asked "why are there no links",
# the same model produced all eight, correct, instantly, saying "my formatting dropped them".
#
# So the payload was never the problem and neither was comprehension: a rule attached to a TOOL
# gets skimmed, while a directive in the session's own setup is the channel that gets obeyed —
# which is exactly what the user's follow-up message proved by working first try.
#
# ⚠️ THIS DOES NOT OVERTURN THE "no prose instructions" RULING BELOW. That ruling stands on
# `instructions` being optional on InitializeResult and droppable by any client, so nothing may
# DEPEND on it. This rule is reinforcement, not load-bearing: `search`'s description still
# carries it for clients that drop this, and every hit still ships the link either way. One
# paragraph, because a blob here competes with the user's own instructions for the same budget.
_CITATION_RULE = (
    "CITING OPYT RESULTS: every knowledge-base item you name in an answer must carry its source "
    "link. Each `search` hit ships a ready-made markdown link in `cite` — write the item's name "
    "inside it, so the title you were already going to write IS the link, costing no extra width. "
    "Never hand back a list of results with no links in it, and never make the user ask where "
    "something came from: a result they cannot click back to is a claim they cannot check."
)


def _server_instructions() -> str | None:
    """Everything this server says at connection setup: the standing citation rule, plus the
    frontier's state when there is any. `None` rather than `""` when there is nothing to say —
    an empty `instructions` is a field a client must decide about for no reason."""
    return (_CITATION_RULE + _frontier_session_notice()).strip() or None


def _diagnostic(message: str) -> None:
    """Write startup diagnostics without corrupting the stdio JSON-RPC transport."""
    print(message, file=sys.stderr, flush=True)


def _setup_client_mcp() -> FastMCP:
    # No prose `instructions` BLOB: `instructions` is optional on InitializeResult and a client MAY
    # drop it, so routing policy belongs in tool descriptions instead. Two things ride here, both
    # of which survive being dropped — `_frontier_session_notice()`, and the standing citation
    # rule (`_CITATION_RULE`, ruled 2026-09-16 on a measured tool-description failure).
    mcp = FastMCP(
        "Opyt",
        instructions=_server_instructions(),
        icons=[Icon(
            src="https://mcp.useopyt.com/icon.png",
            mimeType="image/png",
            sizes=["512x512"],
        )],
    )

    # Every registration below is wrapped so a stripped distribution (e.g. one without
    # `pipeline.kb`) still starts with whatever surface it can offer, rather than failing to boot.

    # Atom-KB "trusted router" tools (search / open / aggregate). LLM-free; the host reasons.
    try:
        from mcp_server.atoms_tools import register_atoms_tools
        register_atoms_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[atoms_tools] registration skipped: {_e}")

    # Oracle SCREEN — onboarding "pick your people" tool (screen / candidates / confirm / ingest).
    try:
        from mcp_server.oracle_tools import register_oracle_tools
        register_oracle_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[oracle_tools] registration skipped: {_e}")

    # `onboard` — thin orchestrator that runs setup stages 2-5 in order for a user.
    try:
        from mcp_server.onboard_tools import register_onboard_tools
        register_onboard_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[onboard_tools] registration skipped: {_e}")

    # FRONTIER — host-judged capture of recent research papers on standing topics; no scoring.
    try:
        from mcp_server.frontier_tools import register_frontier_tools
        register_frontier_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[frontier_tools] registration skipped: {_e}")

    # HOPPER — the one manual "keep this" surface: any URL to an atom. Preview then confirm.
    try:
        from mcp_server.hopper_tools import register_hopper_tools
        register_hopper_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[hopper_tools] registration skipped: {_e}")

    # FORGET — remove one atom or end one Oracle subscription, after a consequence preview.
    try:
        from mcp_server.forget_tools import register_forget_tools
        register_forget_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[forget_tools] registration skipped: {_e}")

    # SITTING — reads one whole topical region of the KB, in publication order, no time bound.
    try:
        from mcp_server.sitting_tools import register_sitting_tools
        register_sitting_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[sitting_tools] registration skipped: {_e}")

    # SHARE / ACCEPT / UNSHARE — knowledge-base sharing, the four moments a person touches.
    try:
        from mcp_server.share_tools import register_share_tools
        register_share_tools(mcp)
    except Exception as _e:
        _diagnostic(f"[share_tools] registration skipped: {_e}")

    return mcp


# ===========================================================================
# Entry point
# ===========================================================================

def _http_port(argv: list[str]) -> int | None:
    """`--http <port>` → the port; no flag → None (stdio). Exits on a malformed value.

    Deliberately hand-rolled rather than argparse: this process's stdout is the JSON-RPC
    channel under stdio, and argparse's error path writes there and raises SystemExit with
    a usage block. One flag does not justify that risk.
    """
    if "--http" not in argv:
        return None
    i = argv.index("--http")
    raw = argv[i + 1] if i + 1 < len(argv) else ""
    if not raw.isdigit() or not (1 <= int(raw) <= 65535):
        sys.exit(f"--http needs a port in 1..65535, got {raw!r}")
    return int(raw)


def main() -> None:
    """Console entrypoint (`opyt-mcp`) and `python mcp_server/server.py`. In-process
    FastMCP. No daemon. Retrieval is served by the SQLite-backed tools (search/etc.) that
    read `$OPYT_HOME/opyt.db`.

    Two modes, one process:

      • **no args** — stdio, the local install.
      • **`--http <port>`** — Streamable HTTP on 127.0.0.1 only, which is a HOSTED CHILD
        spawned by `gateway/`. It serves ONE user, named by the `$OPYT_HOME` its parent set,
        and it authenticates nobody: the gateway did that before proxying, and the bind
        address is what keeps anyone else out.

    **Neither mode launches a rail, and that is load-bearing.** Both modes are disposable
    processes — a stdio server dies with its MCP client, a hosted child is SIGTERMed by the
    gateway's reaper after a few idle minutes — so a rail either mode started would be killed
    mid-ingest, or outlive its parent with nothing tracking it. `opyt-worker` is a resident
    process that outlives both and records every child's exit code. Design records:
    docs/plans/2026-09-06-persistent-rail-worker-migration.md and
    docs/plans/2026-09-02-hosted-opyt-remote-connector.md (R6).
    """
    # First-run bootstrap: write a user-local settings.yaml from the shipped template when none
    # exists yet. No-op on a populated install. Runs in BOTH modes — a hosted child's home is
    # empty the first time the gateway spawns it.
    try:
        from opyt_core.bootstrap import ensure_initialized
        ensure_initialized()
    except Exception as e:
        _diagnostic(f"[bootstrap] skipped: {e}")

    port = _http_port(sys.argv[1:])
    if port is not None:
        # 127.0.0.1, never "localhost": this Mac resolves that to ::1 first, and a child
        # listening only on IPv4 while the gateway dials IPv6 hangs until the spawn deadline.
        mcp = _setup_client_mcp()
        from opyt_core import openrouter_oauth
        from pipeline.ingestion import hosted_browser
        if hosted_browser.enabled() or openrouter_oauth.hosted_enabled():
            # Hosted-only endpoints are loopback relays: the browser boundary owns its Chrome
            # profile and OpenRouter owns its PKCE verifier. The gateway remains their public
            # router and the FastMCP surface stays exactly `/mcp`. Request paths contain one-use
            # callback capabilities, so neither the child nor gateway may access-log them.
            import uvicorn
            hosted_app = mcp.http_app(path="/mcp")
            if hosted_browser.enabled():
                hosted_app = hosted_browser.hosted_child_app(hosted_app)
            if openrouter_oauth.hosted_enabled():
                hosted_app = openrouter_oauth.hosted_child_app(hosted_app)
            uvicorn.run(hosted_app, host="127.0.0.1", port=port, access_log=False)
        else:
            mcp.run(transport="http", host="127.0.0.1", port=port)
        return

    _setup_client_mcp().run()


if __name__ == "__main__":
    main()
