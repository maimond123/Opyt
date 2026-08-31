"""
OPYT pipeline — a NAMESPACE, deliberately. Not a curated public API.

Import submodules directly:

    from pipeline import llm_client
    from pipeline.kb import oracles
    from pipeline.config import state_paths

The re-export facade was deleted here 2026-08-16. It listed 12 symbols (`VaultConfig`,
`get_config`, `check_prerequisites`, `store_credential`, the `types` dataclasses, …) so a caller
could write `from pipeline import get_config`. Verified before removal: NOTHING imported a single
one of them that way — not the MCP server, not a rail, not a test, not a baseline script. All 666
`from pipeline.` imports in this repo reach for a SUBMODULE, which is what
`from pipeline import llm_client` already does and which this file has no say over. Removing the
re-exports therefore changed no import anywhere.

Why it was removed rather than left as harmless clutter. A facade's whole job is to draw a line
between SUPPORTED and INTERNAL, and every benefit of that line accrues to someone OUTSIDE this
repo. No such importer exists, so the line carried no benefit — and it was not free. It laundered
references: a caller-based audit ("who uses this?") hit the export line and read it as a consumer.
That is exactly how `lifecycle.estimate_ingestion_cost` survived FOUR retirement sweeps with zero
callers, and was then repaired and given tests by a fifth reader before anyone asked whether it
should exist. Measured, not theorised: five readers, five misses, one wasted repair.

Cost was not the reason, and the obvious argument is wrong here. Measured 2026-08-16:
`import pipeline` took 55 ms against 186 ms for `from pipeline.kb import oracles` — the facade was
the LIGHTER import, because the heavy modules (kb, embeddings) were never re-exported. The reason
is reference laundering, plus 24 lines declaring 12 names twice.

Do not re-add one casually — and if you do, enforce it. A facade is coherent under exactly two
rules, and this one followed neither:
  • ALL internal code imports through it — then the exports have real callers and grep tells the
    truth, because the boundary is leaned on daily; or
  • NOTHING internal imports through it, and it is a published contract for third parties — then
    "no internal caller" is the expected state rather than evidence of death.
An unenforced facade collects an interface's costs and neither rule's benefit. That is what this
was for its entire life.

When this decision flips: if `pipeline` ever becomes something a third party pip-installs and
imports, a facade is not overhead — it is the product's contract, and it should come back under the
second rule above. Its twin rode the same question: `lifecycle.check_prerequisites` (also
caller-less) was ruled dead weight on 2026-08-29 and deleted with `pipeline/types.py` (see the
`retired-lifecycle-preflight` guard). If the distribution bet ever flips, a revived facade brings
its own preflight on its own terms.
"""
