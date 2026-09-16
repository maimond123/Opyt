"""
pipeline/kb/sitting_claims.py — read one assembled sitting, extract its falsifiable claims.

Sibling to `sitting_reader.py`'s `queries` lens, not a mode of it: separate module, separate
prompt, separate output table (`reader_core.py` holds what they share). The two lenses read
independently — see `sitting_store.mark_lens_read`/`lens_read_state` (`sitting_reads`, keyed
`(sitting_id, lens)`); neither touches `sittings.read_at`, which stays `queries`' own stamp.
`sitting_scheduler`'s retry lane only watches `queries` runs, not `claims`.

The `falsified_by` field is read by a HUMAN, not re-checked by any machine — do not add
"so it can be verified later" near this code; no such consumer exists.

Never raises. Every failure writes NOTHING to `sitting_claims`, leaves the sitting unread for this
lens, and records a `failed` run — same contract as `sitting_reader.read_sitting`.
"""

from __future__ import annotations

from datetime import datetime
from pipeline.timeparse import utc_iso, utc_now

from pipeline.ingestion.utils import log

from . import frontier_queries as fq
from . import reader_core as core
from . import schema
from . import sitting_builder as sb
from . import sitting_reader as sr
from . import sitting_render as sre
from . import sitting_store as sst

LENS = "claims"

# Settled prompt, copied VERBATIM from docs/plans/2026-08-16-sitting-rail-model-bakeoff.md — do
# not tune it here. "Describe what the MATERIAL SAYS. Do not forecast." is load-bearing; do not
# soften it back toward "future" (see the doc for why).
_SYSTEM = """You are reading one SITTING from David's knowledge base — every saved post, thread and
article the system found on a single topic, in chronological order. Read ALL of it before you write
anything. The sitting opens with a header, then:

    ### YYYY-MM-DD — who_id  (atom_id)
    <full text>

Extract the CLAIMS this material makes. A claim is falsifiable when an observation could show it to
be wrong.

Rules:
- 8 to 15 claims. Each one stands alone and is checkable.
- NAME things: systems, companies, standards, numbers, dates. "Adoption is growing" is not a claim.
  "ERC-8004 registered 97,713 agents across 45 chains by mid-March 2026" is.
- Describe what the MATERIAL SAYS. Do not forecast.
- State what would FALSIFY it: the observation that would show the claim wrong.
- CITE EVERY ATOM THAT SUPPORTS THE CLAIM, not just the clearest one. A claim the
  material supports in three places is stronger than one supported once, and the extra citations are
  what make it checkable. Never add a citation that does not support the claim.
- THIS MATERIAL IS ONE CONVERSATION OVER TIME. The strongest claims JOIN A DATE TO A
  LATER ONE: something asserted early and then confirmed, revised, quantified or quietly dropped
  later. Prefer those over claims drawn from a single moment.
- Where the material CONTRADICTS ITSELF over time, say so in the claim and cite BOTH sides.
- AT LEAST HALF YOUR CLAIMS MUST CITE ATOMS MORE THAN A MONTH APART. Three posts from
  the same week describe ONE EVENT; they do not trace a thread. The dates are in the headings —
  check them before you decide a claim is finished.
- Cite the exact bracketed atom_ids that support the claim, copied verbatim from inside
  the ( ) after each date/author heading. The id INCLUDES its source prefix.

Worked example of the shape to aim for. Weak, drawn from one post:
  "ERC-8004 launched on Ethereum mainnet in late January 2026."  [1 citation]
Strong, joining a promise to its later test:
  "Stripe has demonstrated an agent autonomously buying coffee since July 2025, but as of April 2026
   no such agent had shipped to production, and scaled retailers were actively blocking third-party
   agents."  [3 citations, spanning 9 months]
The second is better because a reader can check it, and because it says what CHANGED.

Return ONE JSON object, nothing else:
{"claims": [{"claim": "...", "falsified_by": "...", "atom_ids": ["..."]}]}
"""


def read_claims(conn=None, sitting_id: str = "", *, dry_run: bool = False,
                prompt_only: bool = False, now: datetime | None = None) -> dict:
    """Read one sitting and extract its claims. Never raises.

    `prompt_only` renders the exact prompt and returns without calling anything — the $0 look at
    what would be sent, same contract as `sitting_reader.read_sitting`'s flag of the same name.
    `dry_run` makes the call and returns the result but writes nothing, leaving the sitting unread
    for this lens.

    A `force` here bypassed `_read`'s once-only guard and was deleted on 2026-09-08, unused by
    every caller it ever had. Do NOT bring it back to recover a failed read: the guard it
    bypassed keys on a `sitting_reads` row that only a SUCCESSFUL read writes, so there is
    nothing for it to bypass after a failure, and calling this function is already the retry.
    Two docstrings below promised `--force` as the recovery path for a deterministic failure;
    both were wrong about the mechanism, not merely stale about the CLI that carried the flag.
    """
    ref = now or utc_now()
    own = conn is None
    if own:
        conn = schema.connect()
    try:
        return _read(conn, sitting_id, dry_run=dry_run, prompt_only=prompt_only, ref=ref)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[sitting-claims] run errored: {detail}")
        try:
            fq.record_run(conn, generator=f"{sr.GENERATOR_PREFIX}:?", sitting_id=sitting_id or None,
                          lens=LENS, status="failed", reason=detail)
        except Exception as e2:
            # `_fail` logs this exact failure of this exact call; staying silent here meant a store
            # that cannot write run rows AT ALL left no trace anywhere.
            log(f"[sitting-claims] could not record failed run: {e2}")
        return {"status": "failed", "reason": detail}
    finally:
        if own:
            conn.close()


def collect_notebook_debt(conn, sitting_id: str, *, now: datetime | None = None) -> list[dict]:
    """Pay down every ancestor part that was READ but never had its claims extracted. Never raises.

    THE INVARIANT: the queries read closes a part; the claims receipt is a DEBT every chain walk
    collects. Splitting it that way is what keeps a claims failure from blocking a part —
    `sitting_reader` stamps `read_at` on its own success and this runs afterwards, so a bad claims
    call leaves a notebook entry unresolved, not the part.

    An ancestor with `sittings.read_at` set and no ok `sitting_reads(sitting_id, 'claims')` row owes
    its entry. That state is REACHABLE three ways: a claims call that failed before spending, a
    claims call that failed after spending, and every part read before the notebook existed.
    Collecting here means the debt is paid at the moment it starts to matter — just before part N+1
    is rendered and would otherwise show a hole in its own memory.

    Only the transient ones are collected HERE. A failure that REACHED THE MODEL is dropped by
    `_debtors` — see `_called_and_failed` — so nothing pays that hole down on its own. It is
    retried by asking for the sitting BY NAME: `sitting(action='read', lens='claims',
    sitting_id=…)` reaches `read_claims` directly, and nothing gates it, because `_read`'s
    once-only guard keys on a `sitting_reads` row that the failed read never wrote.

    `read_claims` carries its own once-only guard, so calling it for an already-extracted ancestor
    is a free `skipped`. Returns one result per ancestor it attempted.
    """
    out = []
    for sid in _debtors(conn, sitting_id):
        res = read_claims(conn, sid, now=now)
        if res.get("status") != "ok":
            # Fail-safe: an ancestor that cannot be distilled leaves a hole in the notebook and
            # nothing else. The part about to be read still gets every entry that DOES exist.
            log(f"[sitting-claims] notebook debt on {sid} unpaid: {res.get('reason')}")
        out.append({"sitting_id": sid, **res})
    return out


def read_part(conn, sitting_id: str, *, now: datetime | None = None) -> dict:
    """Close one part: pay the notebook debt, run the `queries` read, then take the claims receipt.
    Returns the `queries` read result. Never raises — every step it calls is fail-safe.

    ONE RITUAL, called by both doors (the `sitting` tool and `sitting_scheduler`), because the
    ORDER is the invariant and a hand-copied order drifts. The notebook is collected BEFORE the
    part is rendered, so the preamble carries what earlier parts established; the claims receipt is
    taken AFTER, so a claims failure costs a notebook entry rather than the part. Wiring only one
    of the two doors would make the notebook depend on who triggered the read.
    """
    collect_notebook_debt(conn, sitting_id, now=now)
    res = sr.read_sitting(conn, sitting_id, now=now)
    if res.get("status") == "ok":
        read_claims(conn, sitting_id, now=now)
    return res


def _debtors(conn, sitting_id: str) -> list[str]:
    """Ancestor parts owing a notebook entry, oldest first."""
    out = []
    for sid in sst.ancestors(conn, sitting_id):
        row = conn.execute("SELECT read_at FROM sittings WHERE sitting_id = ?", (sid,)).fetchone()
        if row is None or not row["read_at"]:
            continue                       # never read → owes nothing; it is not part of the story
        # Presence is the whole test, and since 2026-09-06 there is nothing else to test: this
        # read `state.get("read_status") != "ok"` as well until 2026-09-04, which disagreed with
        # `_read` below — that treats ANY row as already read and skips. The column went with
        # `schema._drop_read_status`, so the two cannot drift apart again.
        if sst.lens_read_state(conn, sid, LENS) is not None:
            continue
        if _called_and_failed(conn, sid):
            continue
        out.append(sid)
    return out


def _called_and_failed(conn, sitting_id: str) -> bool:
    """True when a `claims` read of this sitting reached the model and still produced nothing.

    The failure-side twin of `_read`'s once-only guard, which keys on a `sitting_reads` row a
    failed read never writes — so without this, every later part re-attempted every failed
    ancestor at cost, O(N^2) over a chain.

    `model IS NOT NULL` is exactly the two `_fail(usage=...)` sites: an unparseable body and a
    valid-but-empty one. Both are DETERMINISTIC — the same input and model produce them again — so
    an automatic retry buys nothing — which is the whole reason this check exists, since a
    changed prompt or a changed model is what makes retrying one worthwhile, and neither is
    something a walk of the chain can detect. Everything else stays collectable because it is
    transient: degrade-open, HTTP 402, and a transport exception all record no `model`. Naming
    the sitting on the tool retries either kind; see `collect_notebook_debt` for why that path
    is not gated.
    """
    row = conn.execute(
        "SELECT 1 FROM frontier_reader_runs WHERE sitting_id = ? AND lens = ? "
        "AND status = 'failed' AND model IS NOT NULL LIMIT 1", (sitting_id, LENS)).fetchone()
    return row is not None


def _read(conn, sitting_id: str, *, dry_run: bool, prompt_only: bool,
          ref: datetime) -> dict:
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        return {"status": "failed", "reason": f"no sitting {sitting_id!r}"}
    gen = sr.generator_for(s["seed_ref"])

    # 1. Never re-read what this lens already read. Independent of `queries`' own read state —
    #    see the module docstring for why the two must not gate each other. This fires on
    #    SUCCESS ONLY: `mark_lens_read` is the sole writer of that row and a failed read never
    #    reaches it, so a failure is retried by calling in again and this never sees it.
    state = sst.lens_read_state(conn, sitting_id, LENS)
    if state:
        reason = f"already read at {state['read_at']}"
        fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens=LENS, status="skipped",
                      reason=reason, ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": reason}
    if s["atoms"] == 0:
        reason = "empty sitting — nothing to read"
        fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens=LENS, status="skipped",
                      reason=reason, ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": reason}

    # No standing list — `claims` has nothing analogous to `queries`' survivors to show the model.
    user_msg = sr.render_prompt(conn, sitting_id)

    if prompt_only:
        return {"status": "prompt-only", "sitting_id": sitting_id, "generator": gen,
                "atoms": s["atoms"], "chars": len(user_msg), "est_tokens": len(user_msg) // 4,
                "prompt": user_msg}

    backend = core.resolve_backend()
    reason = core.preflight(backend)
    if reason:
        log(f"[sitting-claims] skipped (degrade-open): {reason}")
        return _fail(conn, gen, sitting_id, ref, reason)

    try:
        resp = core.call(backend, _SYSTEM, user_msg)
    except Exception as e:
        if getattr(e, "status", None) == 402:
            reason = (f"provider rejected the prompt (HTTP 402). Lower "
                      f"OPYT_SITTING_MAX_INPUT_CHARS (currently {sr.MAX_INPUT_CHARS}), "
                      f"Upstream: {e}")
            log(f"[sitting-claims] {reason}")
            return _fail(conn, gen, sitting_id, ref, reason)
        return _fail(conn, gen, sitting_id, ref, f"{type(e).__name__}: {e}")

    usage = {"model": resp.model, "in_tokens": resp.input_tokens,
             "out_tokens": resp.output_tokens,
             "atoms_read": s["atoms"]}
    obj = core.parse_response(resp.text)
    if obj is None:
        reason = "unparseable response body"
        if core.finish_reason(resp) == "length":
            reason = (f"response truncated at max_tokens ({usage['out_tokens']} out) — "
                      f"raise max_tokens for role {core.ROLE!r}")
        return _fail(conn, gen, sitting_id, ref, reason, usage=usage)

    # THE CITATION GATE IS WIDENED TO THE WHOLE CHAIN (decided 2026-08-25). The notebook preamble
    # asks this read to confirm, revise or refute claims made by EARLIER parts, and the honest way
    # to refute one is to cite the ancestor atom it rested on alongside the atom that overturns it.
    # Gated on the shown set alone, `validate_claims` would drop exactly those citations and the
    # notebook could only ever be added to, never corrected.
    #
    # Rows stay append-only: the preamble states that a later row supersedes an earlier one where
    # they conflict. A supersede POINTER column was rejected as premature schema — nothing reads it
    # yet, and the instruction already carries the semantics.
    known = set(sb.chain_atom_ids(conn, sitting_id))
    claims, notes = core.validate_claims(obj, known_atom_ids=known)

    # Same standing signal `queries` writes on every read — the render is chronological regardless
    # of which lens is reading it, so lost-in-the-middle is a shared failure mode (D22).
    chrono = sre.chronological_order(conn, sitting_id)
    cov = core.positional_coverage(chrono["order"], [a for c in claims for a in c["atom_ids"]],
                                   undated=chrono["undated"])
    if cov["note"]:
        notes.append(f"coverage: {cov['note']}")
    for n in notes:
        log(f"[sitting-claims] {n}")

    if not claims:
        # An explicit reject on zero claims: a well-formed but empty response can pass parsing
        # and finish_reason checks (see bakeoff doc for the incident).
        return _fail(conn, gen, sitting_id, ref, "no valid claims after validation", usage=usage)

    if dry_run:
        return {"status": "dry-run", "sitting_id": sitting_id, "generator": gen,
                "claims": claims, "notes": notes, "coverage": cov, **usage}

    sst.record_claims(conn, sitting_id, claims, at=ref)
    fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens=LENS, status="ok",
                  ran_at=utc_iso(ref), emitted=len(claims), middle_share=cov["middle_share"],
                  reason="; ".join(notes) or None, **usage)
    sst.mark_lens_read(conn, sitting_id, LENS, at=ref)
    return {"status": "ok", "sitting_id": sitting_id, "generator": gen, "claims": claims,
            "notes": notes, "coverage": cov, **usage}


def _fail(conn, generator: str, sitting_id: str, ref: datetime, reason: str, *,
          usage: dict | None = None) -> dict:
    """Record a failed run. Writes no claims and leaves the sitting unread for this lens."""
    fields = dict(usage or {})
    try:
        fq.record_run(conn, generator=generator, sitting_id=sitting_id, lens=LENS,
                      status="failed", reason=reason, ran_at=utc_iso(ref), **fields)
    except Exception as e:
        log(f"[sitting-claims] could not record failed run: {e}")
    return {"status": "failed", "reason": reason}
