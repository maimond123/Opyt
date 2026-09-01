#!/usr/bin/env python3
"""
OPYT Knowledge Base MCP Server

The registration point for every tool module, and nothing else — this module defines no
tools of its own. Retrieval is served by `atoms_tools` (`search`, `open`, `aggregate`) over
the atom KB (atoms + chunks + chunks_fts in ~/.opyt/opyt.db).

`main()` also fires each background rail's session-open spawner. Every rail owns its own
spawner in its own try/except, so one rail failing can never stop the server from serving or
take another rail down with it — the convention the `atom-rail-not-welded-to-catchup` guard
enforces.

Usage:
  python mcp_server/server.py      (or the `opyt-mcp` console script)
"""

import sys
from pathlib import Path

from fastmcp import FastMCP

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


def _setup_client_mcp() -> FastMCP:
    # No prose `instructions` blob: `instructions` is optional on InitializeResult and a client MAY
    # drop it, so routing policy belongs in tool descriptions instead. Only
    # `_frontier_session_notice()` rides here.
    mcp = FastMCP(
        "Opyt",
        instructions=_frontier_session_notice().lstrip("\n") or None,
    )

    # Every registration below is wrapped so a stripped distribution (e.g. one without
    # `pipeline.kb`) still starts with whatever surface it can offer, rather than failing to boot.

    # Atom-KB "trusted router" tools (search / open / aggregate). LLM-free; the host reasons.
    try:
        from mcp_server.atoms_tools import register_atoms_tools
        register_atoms_tools(mcp)
    except Exception as _e:
        print(f"[atoms_tools] registration skipped: {_e}", flush=True)

    # Oracle SCREEN — onboarding "pick your people" tool (screen / candidates / confirm / ingest).
    try:
        from mcp_server.oracle_tools import register_oracle_tools
        register_oracle_tools(mcp)
    except Exception as _e:
        print(f"[oracle_tools] registration skipped: {_e}", flush=True)

    # `onboard` — thin orchestrator that runs setup stages 2-5 in order for a user.
    try:
        from mcp_server.onboard_tools import register_onboard_tools
        register_onboard_tools(mcp)
    except Exception as _e:
        print(f"[onboard_tools] registration skipped: {_e}", flush=True)

    # FRONTIER — host-judged capture of recent research papers on standing topics; no scoring.
    try:
        from mcp_server.frontier_tools import register_frontier_tools
        register_frontier_tools(mcp)
    except Exception as _e:
        print(f"[frontier_tools] registration skipped: {_e}", flush=True)

    # HOPPER — the one manual "keep this" surface: any URL to an atom. Preview then confirm.
    try:
        from mcp_server.hopper_tools import register_hopper_tools
        register_hopper_tools(mcp)
    except Exception as _e:
        print(f"[hopper_tools] registration skipped: {_e}", flush=True)

    # SITTING — reads one whole topical region of the KB, in publication order, no time bound.
    try:
        from mcp_server.sitting_tools import register_sitting_tools
        register_sitting_tools(mcp)
    except Exception as _e:
        print(f"[sitting_tools] registration skipped: {_e}", flush=True)

    # SHARE / ACCEPT / UNSHARE — knowledge-base sharing, the four moments a person touches.
    try:
        from mcp_server.share_tools import register_share_tools
        register_share_tools(mcp)
    except Exception as _e:
        print(f"[share_tools] registration skipped: {_e}", flush=True)

    return mcp


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Console entrypoint (`opyt-mcp`) and `python mcp_server/server.py`. In-process
    FastMCP over stdio. No daemon. Retrieval is served by the SQLite-backed tools
    (search/etc.) that read ~/.opyt/opyt.db."""
    # First-run bootstrap: write a user-local settings.yaml from the shipped template when none
    # exists yet. No-op on a populated install.
    try:
        from opyt_core.bootstrap import ensure_initialized
        ensure_initialized()
    except Exception as e:
        print(f"[bootstrap] skipped: {e}", flush=True)

    # ── Session-open rail spawners ──────────────────────────────────────────────────────────
    # Each rail owns its spawner in its own try/except (the `atom-rail-not-welded-to-catchup`
    # guard enforces this) so one rail's bad night can't take another down. Every spawn below
    # coalesces (hourly, mostly) and is single-flight downstream, so this is at most a few ms of
    # fork/exec.

    # The Oracle refresh rail — keeps each confirmed Oracle's sources current.
    try:
        from pipeline.kb.oracle_refresh import spawn_oracle_refresh
        spawn_oracle_refresh()
    except Exception:
        pass  # a trigger failure must never stop the server from serving

    # Frontier stage 2 (EXECUTE) — runs the standing queries. Separate spawner from stage 3
    # because the two fail for different reasons (paid model call vs. flaky upstream index).
    try:
        from pipeline.kb.frontier_execute import spawn_frontier_execute
        spawn_frontier_execute()
    except Exception:
        pass  # ditto

    # Frontier stage 3 (ADMIT) — candidates become atoms. The only spawner here that writes to
    # `atoms`, so the only one with real spend/RAM exposure; bounded by ADMIT_MAX_PER_RUN.
    try:
        from pipeline.kb.frontier_admit import spawn_frontier_admit
        spawn_frontier_admit()
    except Exception:
        pass  # ditto

    # Bookmark catch-up — the automatic X-bookmark trigger; calls `ingest_x.sync_bookmarks`.
    try:
        from pipeline.kb.bookmark_catchup import spawn_bookmark_catchup
        spawn_bookmark_catchup()
    except Exception:
        pass  # ditto

    # Curation catch-up — refreshes the four curation signals (X Lists, following, likes,
    # Substack subs) with no other trigger. Its own spawner, not welded to the bookmark one: this
    # rail is a free, consent-free cookie-scrape and must not wait on a paid rail's gate.
    try:
        from pipeline.kb.curation_catchup import spawn_curation_catchup
        spawn_curation_catchup()
    except Exception:
        pass  # ditto

    # Candidate probe — pulls what each curation candidate actually writes, so
    # `oracle(action='candidates')` answers from their words, not their bio. Bounded by a daily
    # ceiling, not a per-run one, since nothing caps how many sessions a user opens.
    try:
        from pipeline.kb.probe_catchup import spawn_candidate_probe
        spawn_candidate_probe()
    except Exception:
        pass  # ditto

    # Push catch-up — keeps the SERVED copy of this knowledge base current. The only rail here
    # that buys nothing: it no-ops without an `OPYT_SERVICE_TOKEN`, and even with one it pushes
    # only when somebody has read since the last push AND the store has changed since it.
    try:
        from pipeline.kb.push_catchup import spawn_push_catchup
        spawn_push_catchup()
    except Exception:
        pass  # ditto

    # The SITTING rail's disposer — fires a scheduler, not a single-loop rail, because it ranks
    # four channels competing for the same paid reader. See `sitting_scheduler`'s header.
    try:
        from pipeline.kb.sitting_scheduler import spawn_sitting_scheduler
        spawn_sitting_scheduler()
    except Exception:
        pass  # ditto

    _setup_client_mcp().run()


if __name__ == "__main__":
    main()
