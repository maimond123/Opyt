"""
pipeline/kb/sitting_reader.py — read one assembled sitting, emit its standing queries.

Reads one topical region ("sitting") end to end and is the only rail that spends money in the
seeded loop; `sitting_builder.py` grows the region for free. Emits a short consensus plus 10-25
standing queries with provenance into the shared store.

Generator is scoped per region (`generator = "sitting:<slug>"`), not one shared label, so each
region's re-seed and verdicts only ever move that region's own rows.

This rail also renders a keep/drop verdict on the queries already running for its region — the
only source of the decay signal. A drop slows a query's schedule; it never deletes it, and an
unverdicted query is left untouched, never inferred dead from silence.

A sitting is read at most once; consent lives at the deposit, not at each use. See

Never raises. Every failure writes NOTHING to `frontier_queries`, leaves the sitting UNREAD, and
records a `failed` run.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pipeline.timeparse import utc_iso, utc_now

from pipeline.ingestion.utils import log

from . import frontier_queries as fq
from . import reader_core as core
from . import schema
from . import sitting_render as sre
from . import sitting_store as sst
from . import sitting_vectors as sv

GENERATOR_PREFIX = "sitting"

# No per-day read cap: this is an interactive call the user asked for, and consent lives at the
# deposit — gate spend on money-absent/runaway, never on frequency. Runaway is guarded elsewhere:
# one claim per scheduler run, a 3-failure breaker, and a loud 402 path. See companion doc for the
# known residue (a caller that loops `read_sitting` directly is unguarded).

# Ceiling on one call's input. Guards a moving limit, not the model's context: OpenRouter's
# prompt-token cap derives from the remaining credit balance, so it shrinks as credit is spent.
# See companion doc for a measured example of a prompt going from sendable to 402.
MAX_INPUT_CHARS = int(os.environ.get("OPYT_SITTING_MAX_INPUT_CHARS", 800_000))

# Above this share of one author, the region reads as a build log rather than a conversation; the
# prompt is told the number so it can aim past it. See companion doc for the measured example.
SINGLE_AUTHOR_SHARE = 0.60

# Prompt rules below are tuned against measured failures on a real 295-atom window, run three
# times each. Do not edit a rule without re-running that measurement — see companion doc for the
# v0-v3 history of what each rule fixed.
_SYSTEM = """You are reading one SITTING from David's knowledge base — every saved post, thread and
article the system found on a single topic, in chronological order. Read ALL of it before you write
anything. The sitting opens with a header (its seed, its size, its author concentration), then:

    ### YYYY-MM-DD — who_id  (atom_id)
    <full text>

STEP 1 — CONSENSUS (5-10 sentences).
This material is ONE conversation over time, so say how it MOVED. What was claimed early, what
happened to that claim later, what reversed, what got quietly abandoned, and what is unresolved
right now. Name the specific systems, people, benchmarks and numbers. Do not write "a range of
topics" or any other summary that would be true of anyone.

STEP 2 — VERDICTS on the queries already running.
When the sitting begins with CURRENT STANDING QUERIES, those are the searches running on a
schedule for this region right now. Render a verdict on EVERY ONE of them. There is no limit
and no budget here — a list of 40 gets 40 verdicts. A query you say nothing about is left
exactly as it is, so silence is not a way to retire anything; it only hides your opinion.

- text: the listed query, copied character for character. This is how the verdict is
  matched to the query, so an improved or re-capitalized version matches nothing.
- verdict: "keep" if this thread is still live in the sitting — new work on it would still
  interest him. "drop" if the thread is finished, answered, or the conversation moved past it.
  You are reading this region in time order, so you can see which threads the later material
  stopped carrying. That is exactly the judgement being asked for.
- A DROP DOES NOT DELETE THE QUERY. It slows it down: a dropped query goes from daily to
  weekly to monthly and then keeps running monthly forever. So a drop is cheap and
  reversible — a later "keep" restores full speed. Judge honestly rather than defensively.
  Keeping a dead thread alive is the more expensive mistake, because it crowds his attention.
- reason: one line. Why this thread is still live, or what closed it.
- atom_ids: for a "keep", the atom_ids in THIS sitting that show the thread is still live.
  Copy them verbatim, including the source prefix.
- If there is no CURRENT STANDING QUERIES section, return an empty verdicts list.

STEP 3 — STANDING QUERIES (10-25).
These run on a schedule against ARTIFACT sources only — papers, repos, models, datasets, filings —
and they pull work published in the last few weeks. They are not for retrieving what he already
saved. They are for catching what comes NEXT in this thread.

Rules:
- LENGTH: 2 to 6 words. This is the rule most often broken, and breaking it makes the query
  useless. These run against keyword indexes that AND their terms together, so every extra
  word cuts recall toward zero. Name ONE specific thing, plus at most a couple of words
  locating it. "Kimi Delta Attention gated DeltaNet channel decay hybrid MLA decode ablation"
  matches nothing that will ever be published; "gated DeltaNet attention" matches. When a
  thread needs several angles, emit several short queries instead of one long one.
- A query NAMES things — a system, method, architecture, benchmark, lab, gene, ticker.
  Never a category. ("muon optimizer convergence" yes; "AI agents" no — a keyword index
  cannot honor a category.)
- A query must be plausibly matchable by the title or abstract of a paper or repo published
  next month. If nothing new could ever match it, it is a bad standing query.
- No two queries may be near-duplicates. Each one covers a distinct thread.
- Prefer a thread's frontier over its foundation. If he saved a result, query what would
  extend, contradict, or benchmark that result — not the result itself.
- WHEN ONE AUTHOR DOMINATES (the header says so), do not query that author's own projects by
  name — he already follows them, so those results tell him nothing. Query the MECHANISM the
  author is wrestling with, abstracted out of their log: the technique, the failure mode, the
  benchmark someone else would publish against.
- OPEN A NEW QUERY ONLY FOR A THREAD STEP 2 DID NOT ALREADY COVER. A query you kept above is
  already running; listing it again here is not how you confirm it — the verdict is. And do
  NOT restate a standing query in different words. The exact string IS the query's identity,
  so a reworded query is a brand-new one, and it silently discards everything already learned
  about the old one, including how far its sources have been searched.
- On the FIRST read of a region there are no standing queries, so all 10-25 are new. That is
  the normal shape of a first read. On a later read, most threads will be covered by verdicts
  and emitting only a couple of new queries — or none — is a correct answer.
- Cite the exact bracketed atom_ids that motivated the query, copied verbatim from inside
  the ( ) after each date/author heading. The id INCLUDES its source prefix — write
  "x:2083317287604641887" or "substack:203644203", never the bare number.
- rationale: one line, why THIS query for THIS person, now.
- target_sources ∈ {__VALID_SOURCES__}. Route by shape: kernel/systems → github+arxiv;
  bio/trial → biorxiv+pubmed+clinicaltrials; market/company → sec_edgar+hackernews.
  Route only to a source that actually indexes the thing. sec_edgar holds filings by
  US-listed companies only — a private or foreign company does not belong there.
  openalex indexes published literature across EVERY discipline, so route to it whenever
  the thread leaves computer science, or when peer-reviewed work would answer it better
  than a preprint. Do not add it to a query already going to arxiv for the same CS topic —
  it indexes arXiv too, and the same paper found twice is not a second result.

Return ONE JSON object, nothing else:
{"consensus": "...",
 "verdicts": [{"text": "...", "verdict": "keep|drop", "reason": "...", "atom_ids": ["..."]}],
 "queries": [{"text": "...", "target_sources": ["..."], "rationale": "...",
              "atom_ids": ["..."]}]}
"""
# The source vocabulary is INTERPOLATED, never restated. `reader_core.parse_response` drops any
# target_source outside `VALID_SOURCES` and tells the model nothing, so a prompt holding its own
# copy of the list fails silently the moment the two disagree — which is what a second
# hand-maintained copy always does eventually. A `.replace` rather than an f-string because the
# JSON skeleton above is full of literal braces.
_SYSTEM = _SYSTEM.replace("__VALID_SOURCES__", ", ".join(sorted(core.VALID_SOURCES)))


def generator_for(seed_ref: str | None) -> str:
    """`sitting:<slug>` — the dormancy scope. See the module docstring for why it is per region.

    Keyed on the SEED, not the `sitting_id`, on purpose. Re-seeding the same phrase next month is a
    new sitting row but the same conversation, and its queries should refresh that region's rows
    rather than start a parallel set that ages the first one out.
    """
    return f"{GENERATOR_PREFIX}:{sre._slug(seed_ref)}"


def record_lens_run(conn, sitting_id: str, lens: str, *, usage: dict,
                    ref: datetime | None = None) -> None:
    """The trace that one PART was mapped under `lens`. Never stamps `read_at` — that belongs to
    the `queries` lens alone (see `_read` step 1). Never raises.

    ⚠️ AMENDED 2026-08-24, and the amendment is the whole framing. This used to be a receipt with
    no cost, because a lens called no model: it handed the host a document and the host read it
    in-session. Under map-reduce the MAP is a real call per uncached part, so `usage` carries the
    model and what it cost — a run row reporting $0 for a call that spent would make the lens rail
    invisible in every spend report there is.

    `usage` is REQUIRED for that reason: a row is written only where a part was actually mapped,
    and `sitting_lenses._map_part` is the one place that happens. The host-side reduce and
    `sprouts` map nothing and write no row at all.
    """
    ref = ref or utc_now()
    s = sst.get_sitting(conn, sitting_id)
    gen = generator_for(s["seed_ref"]) if s else f"{GENERATOR_PREFIX}:?"
    try:
        fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens=lens, status="ok",
                      ran_at=utc_iso(ref), reason=f"lens map call: {lens}", **usage)
    except Exception as e:
        log(f"[sitting-reader] could not record lens run ({lens}): {e}")


def chain_claims(conn, sitting_id: str) -> list[dict]:
    """Every ancestor part's claims, oldest part first: `[{sitting_id, part, claims}, ...]`.

    The region's running notebook. A part with no claims row is skipped rather than listed empty:
    the notebook says what is established, and an empty heading only invites the model to fill it.
    """
    out = []
    for i, sid in enumerate(sst.ancestors(conn, sitting_id), start=1):  # root is part 1
        claims = sst.get_claims(conn, sid)
        if claims:
            out.append({"sitting_id": sid, "part": i, "claims": claims})
    return out


def _notebook(conn, sitting_id: str) -> str:
    """The claims-carry preamble, or `""` when this is the region's first part.

    THE CROSS-PART MEMORY (RULED 2026-08-24). Old parts appear in later reads only as distilled
    claims, never as re-read text — which is what makes an unbounded region readable at a bounded
    price, and what lets part 2 recognise that the audit it is reading GUTS the number part 1
    established. Under the MMR cut those two atoms landed in different parts and neither reader
    could tell the story.

    The confirm/revise/refute instruction lives HERE and not in either lens's `_SYSTEM`: both
    system prompts are pinned verbatim from the model bake-off and must not be tuned.
    """
    parts = chain_claims(conn, sitting_id)
    if not parts:
        return ""
    lines = [
        "ESTABLISHED SO FAR — the running notebook for this region, distilled from the earlier",
        "parts you are NOT being shown. Each row is a claim a previous read of this same region",
        "made, with the observation that would falsify it.",
        "",
        "As you read the material below, CONFIRM, REVISE or REFUTE each row against it. A later row",
        "supersedes an earlier one where the two conflict. Say which row you are answering and why;",
        "silence about a row is not agreement with it.",
        "",
    ]
    for part in parts:
        lines.append(f"From part {part['part']} (`{part['sitting_id']}`):")
        for c in part["claims"]:
            lines.append(f"  - [C{c['claim_id']}] {c['claim']}")
            lines.append(f"        falsified by: {c['falsified_by']}")
            lines.append(f"        cites: {', '.join(c['atom_ids'])}")
        lines.append("")
    return "\n".join(lines)


def render_prompt(conn, sitting_id: str, *, standing: list[str] | None = None) -> str:
    """The exact text the model reads: the running claims notebook, the standing queries for this
    region, then the sitting itself.

    Reads only. The notebook it renders is whatever `sitting_claims` has already recorded — this
    function never pays to fill a gap, which is `sitting_claims.collect_notebook_debt`'s job and
    is called by the two READ sites before they get here.
    """
    body = sre.render_sitting(conn, sitting_id)
    if len(body) > MAX_INPUT_CHARS:
        # Truncate at the end, not the start: the render is chronological, so cutting the head
        # would delete the beginning of the "what moved" arc that is the point of reading it.
        body = body[:MAX_INPUT_CHARS] + "\n\n[TRUNCATED — sitting exceeds the input budget]"
    # Three flat blocks, in the order a reader needs them: what is already established, what is
    # already being watched, then the new material. Nesting the query list inside a "THIS PART"
    # header would say the standing queries are part of the sitting, which they are not.
    head = []
    note = _notebook(conn, sitting_id)
    if note:
        head.append(note)
    if standing:
        listed = "\n".join(f"- {q}" for q in standing)
        head.append(f"CURRENT STANDING QUERIES for this region "
                    f"(re-emit verbatim if the thread is still live):\n{listed}\n")
    if not head:
        return body
    label = "=== THIS PART ===" if note else "=== SITTING ==="
    return "\n".join(head) + f"\n{label}\n\n{body}"


def read_sitting(conn=None, sitting_id: str = "", *, force: bool = False, dry_run: bool = False,
                 prompt_only: bool = False, standing: bool = True,
                 now: datetime | None = None) -> dict:
    """Read one sitting and upsert its queries. Never raises.

    `prompt_only` renders the prompt and returns without calling anything — the $0 look at exactly
    what would be sent (distinct from the `sitting` tool's `preview` action, which reports scope
    and estimated cost, not the actual prompt bytes). `dry_run` makes the call and returns the
    result but writes nothing, leaving the sitting unread and the query set untouched.

    `standing=False` reads the region as if for the first time, hiding queries already running for
    it — used to measure what a sitting alone generates, or when a region has genuinely shrunk and
    the old query set no longer fits its material. Defaults to True: in steady state the churn it
    prevents is the bigger problem, since a reworded query orphans its watermark. See companion doc
    for the measured BCI case this guards against.
    """
    ref = now or utc_now()
    own = conn is None
    if own:
        conn = schema.connect()
    try:
        return _read(conn, sitting_id, force=force, dry_run=dry_run, prompt_only=prompt_only,
                     standing=standing, ref=ref)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[sitting-reader] run errored: {detail}")
        try:
            fq.record_run(conn, generator=f"{GENERATOR_PREFIX}:?", sitting_id=sitting_id or None,
                          lens="queries", status="failed", reason=detail)
        except Exception:
            pass
        return {"status": "failed", "reason": detail}
    finally:
        if own:
            conn.close()


def _read(conn, sitting_id: str, *, force: bool, dry_run: bool, prompt_only: bool,
          ref: datetime, standing: bool = True) -> dict:
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        return {"status": "failed", "reason": f"no sitting {sitting_id!r}"}
    gen = generator_for(s["seed_ref"])

    # 1. Never re-read what was already read. The sitting is a fixed set of atoms over a fixed
    #    corpus state, so a second read is the same input for the same money. Re-seeding the region
    #    (free) is the way to pick up new material, not re-reading a stale assembly.
    if s["read_at"] and not force:
        reason = f"already read at {s['read_at']}"
        fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens="queries", status="skipped",
                      reason=reason, ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": reason}
    if s["atoms"] == 0:
        reason = "empty sitting — nothing to read"
        fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens="queries", status="skipped",
                      reason=reason, ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": reason}

    running = [r["text"] for r in fq.active_queries(conn, generator=gen)] if standing else []
    user_msg = render_prompt(conn, sitting_id, standing=running)

    if prompt_only:
        # No call, no row, no spend. This is the mode that answers "what EXACTLY would be sent" —
        # not "what would this cost", which is the `sitting` tool's `preview` action.
        return {"status": "prompt-only", "sitting_id": sitting_id, "generator": gen,
                "atoms": s["atoms"], "chars": len(user_msg), "est_tokens": len(user_msg) // 4,
                "standing": len(running), "prompt": user_msg}

    # 3. Preflight — DEGRADE-OPEN. No call attempted, so `cost_usd` stays NULL and the day's
    #    allowance is untouched.
    backend = core.resolve_backend()
    reason = core.preflight(backend)
    if reason:
        log(f"[sitting-reader] skipped (degrade-open): {reason}")
        return _fail(conn, gen, sitting_id, ref, reason, spent=False)

    try:
        resp = core.call(backend, _SYSTEM, user_msg)
    except Exception as e:
        if getattr(e, "status", None) == 402:
            # MONEY-ABSENT, not a broken call: rejected before inference, so nothing was spent and
            # the allowance must not be charged. Fails loud because the remedy is a human action.
            reason = (f"OUT OF CREDITS (HTTP 402) — prompt rejected, nothing spent. Add credits, "
                      f"or lower OPYT_SITTING_MAX_INPUT_CHARS (currently {MAX_INPUT_CHARS}). "
                      f"Upstream: {e}")
            log(f"[sitting-reader] {reason}")
            return _fail(conn, gen, sitting_id, ref, reason, spent=False)
        return _fail(conn, gen, sitting_id, ref, f"{type(e).__name__}: {e}", spent=True)

    usage = {"model": resp.model, "in_tokens": resp.input_tokens,
             "out_tokens": resp.output_tokens, "cost_usd": resp.cost_usd,
             "atoms_read": s["atoms"]}
    obj = core.parse_response(resp.text)
    if obj is None:
        reason = "unparseable response body"
        if core.finish_reason(resp) == "length":
            reason = (f"response truncated at max_tokens ({usage['out_tokens']} out) — "
                      f"raise max_tokens for role {core.ROLE!r}")
        return _fail(conn, gen, sitting_id, ref, reason, spent=True, usage=usage)

    # THE PART'S OWN ADMISSIONS, DELIBERATELY NOT THE CHAIN (RULED 2026-08-25, David: "keep it
    # narrow"). The asymmetry with `sitting_claims`, which DOES widen to `chain_atom_ids`, is the
    # decision and not an oversight: the two lenses cite for different reasons. A claim citing an
    # ancestor atom is CORRECTING the notebook entry that rested on it, which is the one thing the
    # notebook exists for. A query citation says "reading this made me ask this" — and part N's
    # reader has not read an ancestor atom, only a distilled line about it in the preamble. Widening
    # here would let a query claim provenance in text nobody in this run saw.
    #
    # Accepted cost, stated so it is not rediscovered as a bug: a query motivated purely by an
    # ancestor claim, citing only that claim's atoms, is dropped. Re-emission is unaffected — a
    # standing query is carried by its VERDICT, which validates against `shown`, not against atoms.
    # The drop lands in `notes` -> `frontier_reader_runs.reason`, not in the read's result.
    known = {a["atom_id"] for a in s["admissions"]}
    # `shown=running` is the provenance check: `_validate_verdicts` drops any decision not in this
    # exact list, so the model can't verdict a query it invented. It also lets `validate` collapse
    # a query that's both kept and re-emitted as new into one sighting. See companion doc.
    consensus, queries, verdicts, notes = core.validate(obj, known_atom_ids=known, shown=running)
    queries, clamped = _clamp_machine_lane(conn, queries, generator=gen)
    if clamped:
        notes.append(f"machine-lane quota {fq.MACHINE_LANE_QUOTA} reached — dropped "
                     f"{len(clamped)}: {'; '.join(clamped)}")

    # Measures where in the sitting the model actually looked. Render order is chronological, so a
    # mid-document skim would weight the read to the oldest/newest material with every other
    # counter still looking healthy; this is the only signal that catches a route-degraded read
    # (OpenRouter can route to a different provider per run). Note joins `notes` below.
    chrono = sre.chronological_order(conn, sitting_id)
    cov = core.positional_coverage(chrono["order"], [a for q in queries for a in q["atom_ids"]],
                                   undated=chrono["undated"])
    if cov["note"]:
        notes.append(f"coverage: {cov['note']}")
    for n in notes:
        log(f"[sitting-reader] {n}")
    if not queries and not verdicts:
        # Neither a new query nor a verdict means the read understood nothing. A settled region
        # with full verdict coverage and no new threads is still a valid, queries-empty read.
        return _fail(conn, gen, sitting_id, ref, "no valid queries or verdicts after validation",
                     spent=True, usage=usage, consensus=consensus)

    if dry_run:
        return {"status": "dry-run", "sitting_id": sitting_id, "generator": gen,
                "consensus": consensus, "queries": queries, "verdicts": verdicts,
                "notes": notes, **usage}

    # 4. Apply the verdicts, upsert what is genuinely new, then stamp the sitting read.
    #
    #    Verdicts first, so the counters move on decisions rather than on a collision. `validate`
    #    has already removed any "new" query that normalizes onto a verdicted row, so the two calls
    #    cannot both claim the same query in one run.
    marks = fq.apply_verdicts(conn, verdicts, generator=gen, now=utc_iso(ref))
    # `votable=True` is required because this reader asks for verdicts: `_sync_speed` takes the
    # MIN over votable claims only, so a non-votable claim's drops would be recorded but excluded
    # from that aggregate — decay silently dead while the counters look healthy. Known residue: a
    # query claimed by two regions decays only once both have been re-read. See companion doc.
    # `label` is what makes `sitting:<slug>` legible later.
    res = fq.upsert_queries(conn, queries, generator=gen, votable=True,
                            label=f"sitting: {s['seed_ref']} (read {utc_iso(ref)[:10]})",
                            now=utc_iso(ref))
    # Derived, not counted separately: the shown list IS the denominator, so anything neither kept
    # nor dropped got no verdict and was left untouched. Recorded because a rising number here
    # means the survival signal is degrading while every other counter sits still and looks healthy.
    unverdicted = max(len(running) - marks["kept"] - marks["dropped"], 0)
    fq.record_run(conn, generator=gen, sitting_id=sitting_id, lens="queries", status="ok",
                  ran_at=utc_iso(ref), consensus=consensus, emitted=len(queries), new=res["new"],
                  refreshed=res["refreshed"], kept=marks["kept"], dropped=marks["dropped"],
                  unverdicted=unverdicted, reason="; ".join(notes) or None, **usage)
    sst.mark_read(conn, sitting_id, status="ok", at=ref)
    return {"status": "ok", "sitting_id": sitting_id, "generator": gen, "consensus": consensus,
            "emitted": len(queries), "new": res["new"], "refreshed": res["refreshed"],
            "kept": marks["kept"], "dropped": marks["dropped"], "unverdicted": unverdicted,
            # The watchlist diff, NAMED rather than counted (RULED 2026-08-25). A count tells the
            # user three questions are new and leaves them unable to judge or drop any of them.
            # Carried on every read; only a USER-triggered read shows it (see sitting_tools).
            "watchlist_diff": {"new": res["new_texts"], "re_emitted": res["refreshed_texts"],
                               "retired": fq.retired_texts(conn, generator=gen)},
            "notes": notes, "coverage": cov, **usage}


def _clamp_machine_lane(conn, queries: list[dict], *, generator: str) -> tuple[list[dict], list[str]]:
    """Stamp each query's lane and enforce the per-region machine-lane quota. Returns
    `(surviving queries, the texts that were clamped)`.

    Runs POST-PARSE, between `core.validate` and `fq.upsert_queries` — the only point that holds
    resolved citations AND the generator. Never in the prompt: a limit the model is merely asked to
    respect is not a limit, and every query here already carries at least one resolvable atom_id,
    so the classification is always decidable from data.

    Lane rule: machine only when EVERY cited atom is machine-found. One human-attested citation
    makes the query human (mixed = human, ruled) — the human material is what motivated it, and
    the quota exists to bound what the crawler can put on the watch list on its own.

    The allowance counts CLAIMS this generator already holds, not rows it originated, and a
    re-emission of a query already claimed costs nothing: charging it a fresh slot would clamp the
    region's own standing set on the very next read and thrash it in and out forever.
    """
    held = fq.machine_lane_claims(conn, generator)
    allowance = fq.MACHINE_LANE_QUOTA - len(held)
    kept: list[dict] = []
    clamped: list[str] = []
    for q in queries:                                   # emission order IS the priority order
        if _is_machine_lane(conn, q["atom_ids"]):
            q["lane"] = fq.LANE_MACHINE
            qid = fq.query_id_for(fq.normalize(q["text"]))
            if qid not in held:
                if allowance <= 0:
                    clamped.append(q["text"])
                    continue
                allowance -= 1
                held.add(qid)
        else:
            q["lane"] = fq.LANE_HUMAN
        kept.append(q)
    return kept, clamped


def _is_machine_lane(conn, atom_ids: list[str]) -> bool:
    """True when NOT ONE of these atoms is human-attested."""
    rows = conn.execute(
        f"SELECT 1 FROM atoms WHERE entry_mode IN ({sv._in_clause(len(sv.HUMAN_ATTESTED))}) "
        f"  AND atom_id IN ({sv._in_clause(len(atom_ids))}) LIMIT 1",
        tuple(sv.HUMAN_ATTESTED) + tuple(atom_ids)).fetchall()
    return not rows


def _fail(conn, generator: str, sitting_id: str, ref: datetime, reason: str, *, spent: bool,
          usage: dict | None = None, consensus: str | None = None) -> dict:
    """Record a failed run. Writes no queries and leaves the sitting UNREAD.

    Leaving it unread is the point: the region still owes a read, so the ledger keeps asking for it
    instead of counting a failure as coverage.
    """
    fields = dict(usage or {})
    if consensus:
        fields.setdefault("consensus", consensus)
    if spent:
        fields.setdefault("cost_usd", 0.0)
    try:
        fq.record_run(conn, generator=generator, sitting_id=sitting_id, lens="queries",
                      status="failed", reason=reason, ran_at=utc_iso(ref), **fields)
    except Exception as e:
        log(f"[sitting-reader] could not record failed run: {e}")
    return {"status": "failed", "reason": reason}


def unread_sittings(conn) -> list[dict]:
    """Built sittings nobody has read yet, biggest region first — the read queue."""
    return [dict(r) for r in conn.execute(
        "SELECT sitting_id, seed_ref, seed_kind, floor, atoms, tokens, stop, built_at "
        "  FROM sittings WHERE read_at IS NULL AND atoms > 0 "
        " ORDER BY atoms DESC, built_at DESC")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read one sitting and emit its standing queries")
    ap.add_argument("--sitting", help="sitting_id to read (omit to list what is unread)")
    ap.add_argument("--show-prompt", action="store_true", dest="prompt_only",
                    help="print the exact prompt and its token estimate; calls nothing, spends "
                         "nothing. (Named apart from the `sitting` tool's `preview` action, which "
                         "reports a region's SCOPE rather than the bytes that would be sent.)")
    ap.add_argument("--dry-run", action="store_true",
                    help="make the call and print the result; write nothing (this DOES spend)")
    ap.add_argument("--force", action="store_true", help="re-read a sitting already read")
    ap.add_argument("--no-standing", action="store_true",
                    help="hide this region's running queries — read it as if for the first time "
                         "(use when MEASURING what a sitting generates)")
    args = ap.parse_args(argv)

    conn = schema.connect()
    try:
        if not args.sitting:
            print(json.dumps({"unread_sittings": unread_sittings(conn)}, indent=2, default=str))
            return 0
        res = read_sitting(conn, args.sitting, force=args.force, dry_run=args.dry_run,
                           prompt_only=args.prompt_only, standing=not args.no_standing)
        if res.get("status") == "prompt-only":
            print(res.pop("prompt"))
            print("\n" + json.dumps(res, indent=2, default=str))
        else:
            print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("status") in {"ok", "dry-run", "skipped", "prompt-only"} else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
