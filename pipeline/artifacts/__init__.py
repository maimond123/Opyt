"""GitHub artifact helpers — what survived the v1 artifact-frontier rail.

This package is a remnant, not a pipeline. The v1 rail it was named for (standing scopes →
engagement-triggered DISCOVER → relevance-gated auto-commit) was DELETED on 2026-08-12: its
`frontier`, `scopes`, `sweep`, `policy` and `taste` modules are gone, and the
`retired-v1-artifact-frontier` guard blocks re-importing them. Frontier now lives on the atom rail
as five stages in `pipeline/kb/frontier_*.py`, keyed on real external ids in SQLite rather than on
JSON files under `$OPYT_HOME` and the existence of vault markdown.

ONE module is left:

  • `github_client.py` — the GitHub REST transport. LIVE, and its liveness no longer depends on
    `save_repo`: `pipeline/kb/frontier_sources.py` builds a `GitHubApiClient` for the v2 rail's
    GitHub adapter, which Frontier stage 2 runs on main. Deleting this breaks stage 2.

`repos.py` went 2026-08-30, and the WAY it went is the part worth keeping. It had been pruned to
one pure predicate, `repo_moved`, and kept across the 2026-08-13 cut on an explicit bet: Frontier
stage 3 (ADMIT) plausibly wants a materiality test, and re-deriving a threshold invites a different
one silently. Its docstring stated the delete-by condition instead of leaving the bet open — "if
stage 3 grows a re-admission check, import this; if it never does, delete this module." Stage 3
then shipped and settled it the other way: `pipeline/kb/frontier_admit.py` never judges quality at
all, by design. The bet lost, the stated condition fired, and the module went. Write the delete-by
condition down when you keep something on a maybe; it is what lets a later reader close it without
re-arguing the original call. (Guard: `retired-v1b-repo-adapter`, now a module-level ban.)

Do not add to this package. New artifact work belongs on the v2 rail.
"""
