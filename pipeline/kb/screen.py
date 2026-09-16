"""
pipeline/kb/screen.py — Stage-4: the Oracle candidate SCREEN.

Takes Stage-3's canonical entities + their curation_signals and produces the pick-your-Oracles
payload: a structurally-RANKED candidate list, a person-vs-rest classification over EVERY
candidate, and a partition into a pre-ticked recommended set + a floor-filled "see all". The MCP
`oracle` tool hands this to the host, which narrates it in chat; the user confirms.

Three layers, each a plain function over the DB (no MCP dependency, so all testable offline):

  rank_candidates(conn)          → ordered [Candidate], GROUPED BY canonical_id (Fork 1 sort).
  classify_kinds(conn, cands)    → LLM person/org/media/project/aggregator over every UNCLASSIFIED
                                    candidate, batched, CACHED on entities.profile, DEGRADE-OPEN
                                    per batch.
  build_screen(conn)             → the full payload (recommended / shown / see-all partition).

Load-bearing invariants (see the Stage-4 plan):
  • Group by canonical_id, NEVER per-platform entity_id — else a cross-platform person is
    double-counted (the whole reason Stage 3 exists).
  • The payload has to FIT. Returning every signal-bearing candidate with its raw signal rows was
    431,004 characters on a live store — past the host's token limit, so the model could not read
    ANY of it without spilling to a file and `jq`-ing it, which hides all 1,100 people rather than
    the 1,040 below the cut. Since 2026-09-14 the card carries the sentence and not the rows
    behind it (`_card`), and the "see all" tail is bounded with the omission REPORTED
    (`build_screen`, `SEE_ALL_TAIL`). Neither bound may ever touch the default view or a pre-tick.
  • Never HIDE and never REORDER by kind — the payload is in rank order, always. The kind label
    reads name + bio only (often a name alone), which is the weakest evidence in the system;
    letting it move someone out of the default view is the closest thing to hiding a real person.
    So the label's ONLY consequence is the pre-tick — reversing the demotion + persons-only floor
    this file shipped with (David, 2026-08-24, when full classification made them live for ~1000
    people instead of 27). Every card still carries its `kind`, so the host can say what it is.
  • PRE-TICK only corroborated persons — a pre-check is us vouching, and a false-positive Oracle
    is expensive (Stage-5 deep-ingest + becomes a trust root).
  • ENDORSEMENT FIRST, WITHIN A TIER — a person the user endorsed outranks anyone they only ever
    read. See `Candidate.sort_key`, which records the 2026-08-23 reversal this replaced. The tier
    exists because a platform with no follow primitive (a paper registry) can never produce an
    endorsement, so a flat order buries every researcher under every follow for a structural
    reason rather than an evidential one. `interleave_tiers` merges them BY RANK, never by score.
  • NOBODY IN THE SCHOLAR TIER IS PRE-TICKED — a pre-tick is us vouching, and the user picked the
    paper, not its author.
  • `count` is a SOFT tiebreak only, third behind endorsement and distinct-signal count. Never a
    count-weighted score: a weighted model has to defend a ratio, a lexicographic one only has to
    defend a category.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from pipeline.timeparse import parse_ts

from . import schema

# The person-endorsement signals — the PRIMARY rank tier: a follow/list/subscribe is the user
# vouching for the PERSON directly; a save/like vouches for a piece of CONTENT (the author is
# inferred). Subscribe and list stay grouped with follow because all three are person-level acts.
#
# The count that used to justify the grouping was measured before the 2026-09-08 Substack rename
# and has now mostly evaporated: 24 people carried an endorsement without a `follow`, of whom 20
# were Substack rows misnamed `subscribe` that are now `follow`. Four remain. The grouping stands
# on the ARGUMENT rather than on that number — a paid Substack subscription and a curated List are
# person-level acts whether or not many people currently hold one without a follow — and the
# subscription collector shipped the same day produces a real `subscribe` population for the
# first time, which is the number to re-measure against.
#
# `coauthor` and `recommended` are the two types that must never join this set, and it is the same
# reason both times: the user performed no act. Somebody they trust did — an Oracle put their name
# next to this person's, or published a recommendation of them. That is evidence worth ranking,
# and ranking it ABOVE a follow the user made by hand would invert what the tier means.
_ENDORSEMENT = frozenset({"follow", "list", "subscribe"})

# The registries a RESEARCHER's entity id comes from. They have no follow primitive — a person
# cannot follow an author on OpenAlex — so a scholar candidate can never earn an endorsement
# signal and would sort below every X follow permanently under `sort_key` alone. `_tiers` fixes
# that by interleaving, never by weakening the endorsement key.
_SCHOLAR_PLATFORMS = frozenset({"scholar", "openalex"})

# The 5-way kind vocabulary (Fork 3). Only 'person' is ACTED on (pre-tick eligibility); the other
# four are reported-but-inert — they name the card, they never move or hide it.
_KINDS = ("person", "org", "media", "project", "aggregator")

# Batch size for the classifier. A body the ceiling cuts fails `json.loads` outright — an
# oversized batch loses EVERY label in it, not just the tail. 100 was originally sized against a
# 1024-token cap for a non-reasoning model (arithmetic:
# docs/plans/2026-08-24-f5-classify-every-candidate-build.md); the role now runs a REASONING
# model whose thinking draws from the same max_tokens budget, so the cap is 16000 in the role
# config and the binding constraint here is no longer token arithmetic. 100 stays anyway: the
# 1-based verdict alignment below gets no safer with bigger batches, and one lost batch should
# cost 100 labels, not 300. The 2026-09-15 incident is the cautionary tale — at the old 1024
# cap, gpt-oss-120b spent 1020 tokens thinking about 100 real bios and returned EMPTY content,
# so 8 of 9 batches died and only the 29-candidate remainder classified.
CLASSIFY_BATCH = 100

# SHOW a floor so a thin user gets a real list rather than being dumped to the free-form box.
DEFAULT_FLOOR = 15
CORROBORATION_MIN = 2          # distinct (type,platform) signals to be "corroborated"


# ── Candidate ────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    canonical_id: str
    name: str | None = None
    handle: str | None = None
    kind: str | None = None                 # classified kind, or None = not-yet-classified
    distinct_signals: int = 0               # # of DISTINCT (signal_type, platform) — rank key b
    has_endorsement: bool = False           # any follow/list/subscribe — the PRIMARY rank key
    total_count: int = 0                    # Σ action count (soft tiebreak only)
    signals: list = field(default_factory=list)      # [{signal_type, platform, count, extra}]
    members: list = field(default_factory=list)      # per-platform entity_ids in this cluster
    identity_links: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)      # {bio, verified, followers, …} for classify
    retired: bool = False                            # unfollowed — see `rank_candidates`

    @property
    def corroborated(self) -> bool:
        return self.distinct_signals >= CORROBORATION_MIN

    @property
    def is_scholar(self) -> bool:
        """Every signal this person carries came from a paper registry.

        ALL, not any: someone the user follows on X who also wrote a paper they saved has a real
        endorsement and belongs in the main tier at the rank that endorsement earns them. Only a
        person the user knows PURELY as an author needs the tier."""
        return bool(self.signals) and all(
            s["platform"] in _SCHOLAR_PLATFORMS for s in self.signals)

    @property
    def is_person(self) -> bool:
        # DEGRADE-OPEN: unclassified (kind is None) is treated as person-ELIGIBLE, so a skipped
        # classify never costs anyone a pre-tick. Only an explicit non-person kind blocks one, and
        # a pre-tick is all this decides — see the module docstring's never-reorder invariant.
        return self.kind in (None, "person")

    def sort_key(self) -> tuple:
        # Descending priority. Negate the DESC numerics; canonical_id ASC last so the "see all"
        # order is STABLE across renders.
        #
        # ENDORSEMENT IS THE PRIMARY KEY, and that is a REVERSAL (David, 2026-08-23). Fork 1
        # originally led with `distinct_signals`, so save+like (two content signals) outranked a
        # lone follow — the old test named that intended, as "revealed preference over a passive
        # follow". Measured on the live store, it put all 31 content-mixed people at 172-202,
        # above all 268 follow-only people starting at 220.
        #
        # The reversal's argument: a follow/list/subscribe is a PERSON-level act, a save/like is a
        # CONTENT-level one, and a person-level act wins categorically however much content
        # accumulates. Tiered, not weighted, so there is no ratio to defend.
        #
        # Inside a tier nothing changed: distinct signals, then count. So save and like still carry
        # identical weight, and variety still beats volume (1 save + 1 like outranks 5 saves) —
        # both deliberate, both ruled in the same conversation.
        #
        # Flips if endorsement-bearing people prove to be stale follows the user never confirms
        # while high-content strangers below them do get confirmed. That is evidence a follow is
        # NOT categorically stronger, and it is the only thing that should reopen this.
        # Design record: docs/plans/2026-08-23-candidate-ranking-endorsement-first.md
        return (not self.has_endorsement, -self.distinct_signals,
                -self.total_count, self.canonical_id)


# ── (a) ranking ────────────────────────────────────────────────────────────────

def _loads(v) -> list | dict:
    if not v:
        return []
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return []


def _as_dict(v) -> dict:
    """A stored JSON field that MUST be a dict (profile/extra) → a dict, defaulting to {} for
    null/empty/malformed. `_loads` defaults to [] (right for list fields like identity_links);
    calling .get/.items on that [] is the trap this closes."""
    d = _loads(v)
    return d if isinstance(d, dict) else {}


def _best_name(members: list[dict]) -> tuple[str | None, str | None]:
    """(display_name, handle) for a cluster. Prefer a member with a non-null name; among ties
    prefer the X row (Fork 1: the signal may sit on substack:carol while the head is x:user:123,
    and the X profile carries the richer name). `handle` comes from the X member's stored profile."""
    x_members = [m for m in members if (m["entity_id"] or "").startswith("x:")]
    ordered = x_members + [m for m in members if m not in x_members]
    name = next((m["name"] for m in ordered if m["name"]), None)
    handle = next((_as_dict(m["profile"]).get("handle") for m in ordered
                   if _as_dict(m["profile"]).get("handle")), None)
    return name, handle


# The one COLLECTOR whose walk is trusted to prove an absence. Retirement compares a signal
# against one walk, so the unit here is a collector, not a signal type — and it names the
# collector key rather than being derived from `signal_type`, because a signal type no longer
# identifies one collector. `substack_follows` also lands `follow` rows as of 2026-09-08, and a
# lookup by signal type silently picked whichever spec came first in the registry, which turned
# retirement off entirely without failing anything.
#
# Only X following. Append-only signals (like/save/subscribe) never veto a retirement, `list` has
# too few rows to be a habit, and the Substack follow walk is not trusted for an ABSENCE: that
# endpoint is Cloudflare-refused often enough that "not seen this pass" is routinely a block
# rather than an unfollow. Adding it is a behaviour change with its own measurement, not a
# side effect of the collector existing.
MAINTAINED_COLLECTOR = "x_following"


def _retired_ids(conn) -> set[str]:
    """Entity ids whose MAINTAINED signal a healthy full walk failed to re-confirm.

    Two gates, and both must pass before absence counts as evidence. The collector's last walk has
    to be trustworthy (`curation_state.walk_is_trustworthy` — an `ok` run whose `found` did not
    collapse), and the signal's own `last_confirmed_at` has to predate that walk.

    Fail-safe is the empty set, and every failure path returns it: no clock, an unreadable
    clock, an untrustworthy walk, an unparseable stamp. Retiring nobody costs a cycle; retiring
    someone a broken walk failed to see costs a signal no later run brings back.

    Those four are handled EXPLICITLY below and the guard is narrowed to the store — a broken
    import or a renamed field is a programming error, not a missing clock, and returning the
    empty set for it made a permanently disabled retirement look exactly like a healthy one."""
    from . import curation_state
    from .ingest_curation import SPEC_BY_COLLECTOR

    spec = SPEC_BY_COLLECTOR[MAINTAINED_COLLECTOR]
    try:
        run = curation_state.get_run(conn, spec.collector)
        if not curation_state.walk_is_trustworthy(run):
            return set()
        # The walk's start, not its finish. A collector confirms people as it goes and stamps the
        # clock at the end, so comparing against `last_ok_at` reads every person a healthy walk saw
        # as unconfirmed — it retires the whole list. A row with no `started_at` is pre-upgrade
        # state and retires nobody, which is the fail-safe direction.
        walked_at = parse_ts(run.started_at)
        if walked_at is None:
            return set()
        # `set_signal` stamps through SQLite's `datetime('now')`, which truncates DOWN to the
        # second. A person confirmed 0.4s into the walk therefore records a whole-second stamp that
        # can precede the walk's own sub-second start. Truncating the boundary the same way makes
        # the two directly comparable, instead of papering over it with a magic margin.
        walked_at = walked_at.replace(microsecond=0)
        out = set()
        for r in conn.execute(
                "SELECT entity_id, last_confirmed_at FROM curation_signals "
                " WHERE signal_type=? AND platform=?", (spec.signal_type, spec.platform)):
            seen = parse_ts(r["last_confirmed_at"])
            if seen is not None and seen < walked_at:
                out.add(r["entity_id"])
        return out
    except sqlite3.Error:          # a store without the retirement tables retires nobody
        return set()


def _is_maintained(row) -> bool:
    """Is this signal row the one the maintained collector writes?

    Matches on signal type AND platform. Type alone would let a Substack `follow` — which no walk
    is ever compared against, so it can never be retired — permanently rescue a cluster whose X
    follow went stale."""
    from .ingest_curation import SPEC_BY_COLLECTOR

    spec = SPEC_BY_COLLECTOR[MAINTAINED_COLLECTOR]
    return row["signal_type"] == spec.signal_type and row["platform"] == spec.platform


def rank_candidates(conn, *, include_retired: bool = False) -> list[Candidate]:
    """Group every curation signal by canonical_id and rank the resulting people structurally
    (Fork 1). Candidate universe = SIGNAL-BEARING canonical entities only (an atom-author with no
    curation signal is corpus, not a candidate). Returns the ordered list; empty when no signals.

    RETIRED people are dropped by default — see `MAINTAINED_COLLECTOR`. `include_retired=True` returns
    them with `.retired` set, which is what lets a caller REPORT the count rather than let them
    silently vanish."""
    rows = schema.signals_with_canonical(conn)
    retired_ids = _retired_ids(conn)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["canonical_id"], []).append(r)

    cands: list[Candidate] = []
    for cid, grp in groups.items():
        distinct = {(r["signal_type"], r["platform"]) for r in grp}
        # One member row per entity_id (its name/links/profile), deduped.
        members: dict[str, dict] = {}
        for r in grp:
            members.setdefault(r["entity_id"], {
                "entity_id": r["entity_id"], "name": r["name"],
                "identity_links": r["identity_links"], "profile": r["profile"]})
        member_rows = list(members.values())
        name, handle = _best_name(member_rows)
        links: list = []
        for m in member_rows:
            for u in _loads(m["identity_links"]):
                if u and u not in links:
                    links.append(u)
        # Merge the member profiles for classify inputs; carry the cached classified_kind if any.
        prof: dict = {}
        for m in member_rows:
            prof.update({k: v for k, v in _as_dict(m["profile"]).items() if v is not None})

        cands.append(Candidate(
            canonical_id=cid, name=name, handle=handle,
            kind=prof.get("classified_kind"),
            distinct_signals=len(distinct),
            has_endorsement=any(st in _ENDORSEMENT for st, _ in distinct),
            total_count=sum(int(r["count"] or 0) for r in grp),
            signals=[{"signal_type": r["signal_type"], "platform": r["platform"],
                      "count": int(r["count"] or 0), "extra": _loads(r["extra"])} for r in grp],
            members=[m["entity_id"] for m in member_rows],
            identity_links=links, profile=prof,
            # A CLUSTER is retired when every one of its members' maintained signals is. A person
            # resolved across two platforms whose X follow is stale but who was re-confirmed under
            # another member id is still followed.
            retired=bool(retired_ids) and all(
                m["entity_id"] in retired_ids for m in member_rows
                if any(_is_maintained(r) and r["entity_id"] == m["entity_id"]
                       for r in grp)) and any(_is_maintained(r) for r in grp),
        ))

    cands.sort(key=Candidate.sort_key)
    return cands if include_retired else [c for c in cands if not c.retired]


def interleave_tiers(ranked: list[Candidate]) -> list[Candidate]:
    """Per-platform tiers, INTERLEAVED BY RANK — never by adding scores.

    `sort_key` makes endorsement the primary key, categorically (David, 2026-08-23). OpenAlex has
    no follow primitive, so a researcher can never produce an endorsement signal, and under one
    flat order every scholar candidate sits below every X follow no matter how many of their
    papers the user saved. That is the ordering being wrong for a structural reason, not the
    endorsement key being wrong.

    Interleaving by RANK POSITION is the fix, and it is the same rule `oracle(action='candidates')`
    already states for its two evidence bases: take the nth of each tier in turn. Adding or
    weighting scores across tiers is the rejected alternative — it needs a ratio between "a follow"
    and "three saved papers" that nothing in the system can defend.

    A single-platform user has one non-empty tier, so this returns their list untouched.
    """
    scholars = [c for c in ranked if c.is_scholar]
    if not scholars or len(scholars) == len(ranked):
        return ranked                     # one tier — nothing to interleave
    main = [c for c in ranked if not c.is_scholar]
    out: list[Candidate] = []
    for i in range(max(len(main), len(scholars))):
        if i < len(main):
            out.append(main[i])
        if i < len(scholars):
            out.append(scholars[i])
    return out


def reflect(cand: Candidate) -> str:
    """Reflect the user's OWN signals back as a short human phrase — "you follow · subscribe ·
    bookmarked 12×". DEGRADES honestly: 'subscribe (paid)' only when is_paid is known True; a
    None/unknown is_paid falls to a plain 'subscribe'.

    That paid branch was dead until 2026-09-08 — the only writer of a Substack `subscribe` signal
    read the FOLLOW endpoint, which carries no payment field, and recorded `is_paid: None` on every
    row. `ingest_curation.sync_substack_subscriptions` now supplies it from `membership_state`, so
    this is the first code path that makes a claim about the user's money. `_is_paid` there holds
    the mapping and states which arm of it is measured and which is inferred."""
    parts: list[str] = []
    for s in cand.signals:
        st, pf, c, extra = s["signal_type"], s["platform"], s["count"], (s["extra"] or {})
        if st == "follow":
            parts.append("you follow")
        elif st == "subscribe":
            parts.append("you subscribe (paid)" if extra.get("is_paid") is True else "you subscribe")
        elif st == "list":
            names = extra.get("list_names") or []
            parts.append(f"in {c} of your Lists" + (f" ({', '.join(names)})" if names else ""))
        elif st == "save":
            # Keyed on the platform, not an `x`/not-`x` binary: the binary read every non-X save
            # as a Substack post, so an author of three saved PAPERS was reflected back as
            # "saved 3 post(s)" — a claim about content the user never saw.
            parts.append({"x": f"bookmarked {c}×",
                          "scholar": f"you saved {c} of their paper(s)",
                          "openalex": f"you saved {c} of their paper(s)",
                          }.get(pf, f"saved {c} post(s)"))
        elif st == "like":
            parts.append(f"liked {c} of their posts")
        elif st == "coauthor":
            # The weakest signal in the system, and the phrasing says so: the user did nothing.
            # An Oracle put their name next to this person's, on `c` papers.
            parts.append(f"co-wrote {c} paper(s) with one of your Oracles")
        elif st == "recommended":
            # The other signal the user did not create. Same phrasing discipline as `coauthor`:
            # name whose act this was, so the candidate is never mistaken for something the user
            # chose. `c` is how many Oracles recommend them, not how many times.
            parts.append("recommended by one of your Oracles" if c <= 1
                         else f"recommended by {c} of your Oracles")
    return " · ".join(parts)


# ── (b) classifier ───────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = (
    "You label X/Substack accounts by what the account IS, for a knowledge-base onboarding step. "
    "Return STRICT JSON mapping each item's number (as a string) to exactly one kind:\n"
    "  person     — an individual human (even if they run a newsletter/company).\n"
    "  org         — a company/organization/lab account.\n"
    "  media       — a publication/outlet/news brand.\n"
    "  project     — a product/protocol/tool/repo account (not a person).\n"
    "  aggregator  — a bot/list/firehose that re-posts many voices (not one voice).\n"
    "Judge from the name + bio + signals. When genuinely unsure, prefer 'person'. "
    'Respond ONLY with the JSON object, e.g. {"1":"person","2":"org"}.'
)


def _classify_prompt(batch: list[Candidate]) -> str:
    lines = []
    for i, c in enumerate(batch, 1):
        p = c.profile or {}
        bio = (p.get("bio") or "").replace("\n", " ")[:280]
        followers = p.get("followers")
        verified = p.get("verified")
        meta = []
        if c.handle:
            meta.append(f"@{c.handle}")
        if verified is not None:
            meta.append("verified" if verified else "unverified")
        if followers is not None:
            meta.append(f"{followers} followers")
        platforms = sorted({s["platform"] for s in c.signals})
        meta.append("on " + "+".join(platforms))
        lines.append(f'{i}. {c.name or "(unknown)"} [{", ".join(meta)}]'
                     + (f" — bio: {bio}" if bio else " — (no bio)"))
    return "Classify each account:\n\n" + "\n".join(lines)


def _parse_verdicts(text: str, n: int) -> dict[int, str]:
    """LLM text → {index: kind}, keeping only valid indices + kinds. Tolerant of the Llama
    fenced/prefixed-JSON habit; a fully unparseable body yields {} (→ degrade-open upstream)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1] or t)
    except (ValueError, TypeError):
        return {}
    out: dict[int, str] = {}
    for k, v in (obj.items() if isinstance(obj, dict) else []):
        try:
            idx = int(str(k).strip())
        except (ValueError, TypeError):
            continue
        kind = str(v).strip().lower()
        if 1 <= idx <= n and kind in _KINDS:
            out[idx] = kind
    return out


CLASSIFY_ROLE = "entity_classify"


def classify_kinds(conn, candidates: list[Candidate]) -> dict:
    """Classify EVERY unclassified candidate's 5-way kind, in batches of `CLASSIFY_BATCH`, caching
    each verdict on its canonical entity (`profile.classified_kind`) and setting it on the in-memory
    Candidate. Idempotent: already-classified candidates are skipped, so a re-screen re-spends only
    on newly-surfaced people — which also makes an interrupted run resume for free.

    No `top_n`: the list you hand it IS the scope. The knob it replaced bounded a cost measured at
    two cents for all 982 unclassified candidates on the live store.

    DEGRADE-OPEN + SKIP-SAFE, PER BATCH: a missing role/key returns before any call; a batch that
    errors or comes back unparseable writes NOTHING for its own hundred and leaves them kind=None
    (person-eligible), while every other batch still lands (`feedback_llm_failure_must_skip`)."""
    from pipeline.ingestion.utils import log

    # Scholars are excluded, not because a researcher is always a person — OpenAlex issues author
    # ids to consortia too ("ATLAS Collaboration") — but because the label has NO CONSEQUENCE for
    # them. `kind` decides exactly one thing, the pre-tick, and no scholar is ever pre-ticked. A
    # paid call whose answer changes nothing is waste, and it also removes the aggregate
    # classification cost of a few hundred coauthors entirely.
    pending = [c for c in candidates if c.kind is None and not c.is_scholar]
    if not pending:
        return {"ran": True, "classified": 0, "note": "every candidate already classified"}

    from pipeline import llm_client

    # preflight: a missing role or absent key degrades OPEN rather than raising into the screen.
    # A global condition, so it is checked ONCE — a missing key should cost one check, not ten.
    try:
        reason = llm_client.preflight(CLASSIFY_ROLE)
    except Exception as e:
        reason = f"role {CLASSIFY_ROLE!r} unavailable: {e}"
    if reason:
        log(f"[screen] classify skipped (degrade-open): {reason}")
        return {"ran": False, "reason": reason, "classified": 0}

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).date().isoformat()
    batches = [pending[i:i + CLASSIFY_BATCH] for i in range(0, len(pending), CLASSIFY_BATCH)]
    classified, failed, first_reason = 0, 0, None

    for batch in batches:
        # ⚠️ ALIGNMENT: `_parse_verdicts` keys verdicts 1-based into THIS batch, and the write
        # indexes back into THIS list object. Never share one index space across batches and never
        # reorder a batch after its call — a misaligned write lands a verdict on the wrong person,
        # and there is no reader downstream that would notice.
        err, verdicts = None, {}
        try:
            resp = llm_client.call(CLASSIFY_ROLE, system=_CLASSIFY_SYSTEM,
                                   user=_classify_prompt(batch))
            verdicts = _parse_verdicts(resp.text, len(batch))
            if not verdicts:
                err = "no parseable verdicts"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if err:
            failed += 1
            first_reason = first_reason or err
            log(f"[screen] classify batch of {len(batch)} skipped (degrade-open): {err}")
            continue
        for idx, kind in verdicts.items():
            cand = batch[idx - 1]
            cand.kind = kind
            # Cache on the CANONICAL entity's row so re-screen skips it. Fail-safe per write.
            try:
                schema.set_entity_profile(conn, cand.canonical_id,
                                          {"classified_kind": kind, "classified_at": stamp})
                classified += 1
            except Exception as e:
                log(f"[screen] cache write failed for {cand.canonical_id}: {e}")

    out = {"ran": classified > 0, "classified": classified, "of": len(pending),
           "batches": len(batches)}
    if failed:
        out["failed_batches"] = failed
        out["reason"] = first_reason
        # ⚠️ BATCH ACCOUNTING IS NOT A USER-FACING FACT. `failed_batches` and `reason` are for
        # whoever is debugging the classify role; they were the only account of the degrade the
        # host could see, so the host read them out. Measured 2026-09-15, to a user choosing who
        # to trust: "Most of the automatic sorting failed: 8 of 9 attempts gave no usable result."
        # Nobody can act on a batch count. What they CAN act on is that the list in front of them
        # is rougher than usual — degrade-open means the unsorted candidates are still shown, and
        # some of them are organisations wearing a person's row. So this says what the degrade
        # costs the reader, and leaves the arithmetic to the log.
        out["host_note"] = (
            "Some candidates could not be sorted into people versus organisations on this pass, "
            "so this list is rougher than usual and a few companies or projects will appear as "
            "people. Say that ONCE, in plain words, as something to watch for while they pick. "
            "Do NOT report batch counts, failure counts or a reason string — none of it is "
            "theirs to act on. Nothing was lost: the unsorted ones are still listed, and a later "
            "pass sorts them.")
    return out


# ── (c) assembly ─────────────────────────────────────────────────────────────────

def _card(cand: Candidate, *, pre_ticked: bool, shown_by_default: bool) -> dict:
    """One candidate as the host reads them — a person, not a row of the store.

    ⚠️ `signals`, `identity_links` and `members` USED TO RIDE HERE AND DO NOT ANY MORE (2026-09-14).
    They were the payload: a live screen measured 431,004 characters and then 459,124, over the
    host's token limit both times, which forced the model to spill the result to a file and `jq`
    it — two of one session's three filesystem excursions started right here
    (docs/plans/2026-09-14-sixty-second-wall.md §D).

    Nothing was lost by removing them, which is why they went rather than being paginated. The
    three fields are the RAW form of things this card already states in the form the host needs:
    `reflected` is `signals` written as a sentence ("you follow · bookmarked 12×"),
    `distinct_signals` is its length, and `canonical_id` is the handle onto the cluster that
    `identity_links` and `members` enumerate. No reader outside this module ever read them off a
    card — checked across the repo — and a payload nobody reads is not detail, it is weight.

    Do not re-add them "for completeness". A `screen` is a list of people to say out loud.
    """
    return {
        "canonical_id": cand.canonical_id,
        "name": cand.name,
        "handle": cand.handle,
        "kind": cand.kind or "unclassified",
        "is_person": cand.is_person,
        # MECHANICAL only, and empty for most candidates. A scholar's card carries one because a
        # user reading a paper usually does not know who wrote it — see `scholar_probe.describe`,
        # which builds it from the user's own saved titles, then stated institution and counts.
        "description": (cand.profile or {}).get("description") or "",
        "corroborated": cand.corroborated,
        "pre_ticked": pre_ticked,
        "shown_by_default": shown_by_default,
        "distinct_signals": cand.distinct_signals,
        "total_count": cand.total_count,
        "reflected": reflect(cand),
    }


# The MOST candidate cards a `screen` returns, in rank order. A TOTAL, not a tail.
#
# ⚠️ IT WAS A TAIL CAP FOR ONE BUILD, AND THAT CAP MISSED THE PAYLOAD ENTIRELY. The first fix
# bounded only the candidates BELOW the default view and let the default view ride whole, on the
# reasoning that it was 15-30 people and that a pre-tick is a vouch too important to cut. Measured
# 2026-09-14 on a real store: `shown_by_default` is `pre_ticked OR within floor`, and `pre_ticked`
# is UNBOUNDED — 179 of 1,070 people cleared it. The payload came back at 68,476 characters, over
# the host's limit exactly as before, and a second call with `floor=1, see_all=0` still returned
# 178 cards. The number that was believed to be 15-30 was the FLOOR, which is a different field.
#
# So the cap is on the whole list now and there is no exemption, because an exemption for an
# unbounded set is not a cap. Nobody is protected by a vouch inside a payload the reader cannot
# open: at 1,070 candidates the old "nothing is hidden" hid all of them.
#
# 40 × ~297 measured chars/card ≈ 12KB, plus ~4KB of lookback/freshness/note ≈ 16KB — comfortably
# inside a limit that 52KB was already past. The host narrates ~15 and mentions a few more; the
# counts (`total_candidates`, `recommended_count`, `omitted`) carry the truth about the rest.
SCREEN_LIMIT = 40
# The HARD ceiling on `limit`, not a suggestion. The `omitted` note used to say "do NOT raise it
# past ~80" and a host raised it to 1100 anyway (measured 2026-09-14: 290,754 characters, past the
# token limit, recovered only by spilling to a file and `jq`-ing it). A documented "do not" that
# the tool cheerfully honors is not a guard. The ask behind that call was "show me my Substack
# people", which is what `source=` now answers in bounds.
SCREEN_LIMIT_MAX = 80


def known_platforms(conn) -> list[str]:
    """Every platform this store actually holds a signal from — the vocabulary `source=` accepts.
    Read from the data, not a constant, so a new collector needs no edit here."""
    return sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT platform FROM curation_signals WHERE platform IS NOT NULL") if r[0])


def _on_platform(cand: Candidate, source: str) -> bool:
    """Does this person reach the user through `source`? Any ONE signal is enough.

    Membership, not exclusivity: Dean W. Ball arrives by X follow AND Substack subscribe, and he
    is a Substack person by any reading the user means when they ask for one. Requiring every
    signal to match would hide exactly the corroborated people a screen exists to surface.
    """
    return any(sig.get("platform") == source for sig in cand.signals)


def build_screen(conn, *, floor: int = DEFAULT_FLOOR, limit: int = SCREEN_LIMIT,
                 source: str | None = None) -> dict:
    """The full SCREEN payload the `oracle` tool hands the host. Ranks, classifies every
    unclassified candidate (degrade-open), then partitions:
      • pre_ticked       = corroborated (distinct≥2) AND person AND not a scholar — the
                           default-YES vouch set.
      • shown_by_default = pre_ticked OR within the visibility floor, in RANK order.
      • the rest ride behind "see all", also in rank order.
    Nothing is reordered by kind — every candidate returned is in `candidates` in rank order, each
    flagged with its `kind` so the host can say what it is. See the module docstring for why the
    label stops at the pre-tick.

    ⚠️ "NOTHING IS HIDDEN" WAS TRUE OF THE PAYLOAD AND FALSE OF THE READER. Every signal-bearing
    candidate used to ride, and on a real store that was 431,004 characters — past the host's
    token limit, so the model could not read ANY of it without spilling to a file first. A list
    that cannot be read hides all 1,100 of its people, not the 1,040 below the cut. So the list is
    bounded and the omission is REPORTED (`omitted`, with the `limit=` that widens it), which is
    the difference between a bound and a silent truncation. The classify still runs over everyone
    — the cap is on what is RETURNED, never on what is considered, so ranks do not move when it
    changes."""
    # REFUSE, NEVER GUESS — and an empty list is a guess here. A misspelled `source` would
    # otherwise return zero candidates, which reads exactly like a platform the user has connected
    # and nobody arrives through. Naming what this store DOES have is the whole repair.
    if source:
        known = known_platforms(conn)
        if source not in known:
            return {"error": f"source={source!r} is not a platform this store holds a signal "
                             f"from. Known: {', '.join(known) or '(none yet)'}. Nothing was "
                             f"filtered; omit `source` for the whole roster.",
                    "known_platforms": known}

    ranked = interleave_tiers(rank_candidates(conn))
    classify = classify_kinds(conn, ranked)

    # Floor filled in RANK order, kind-blind: an org at rank 3 keeps rank 3 and its place in the
    # default view; it just does not arrive pre-ticked.
    floor_ids = {c.canonical_id for c in ranked[:floor]}

    # THE CUT IS ON MEMBERSHIP, NEVER ON RANK. Filtering happens after `rank_candidates` and
    # `floor_ids`, so a matched person keeps the rank and the floor status they hold in the whole
    # list — `source=` narrows who is listed, it never re-scores anybody.
    candidates, recommended = [], 0
    for c in ranked:
        if source and not _on_platform(c, source):
            continue
        # NOBODY in the scholar tier is pre-ticked. A pre-tick is OPYT vouching, and OPYT should
        # not vouch for someone it inferred from an author list — the user picked the paper, not
        # the person. They render unchecked, in rank order, with their description to judge by.
        pre = c.corroborated and c.is_person and not c.is_scholar
        if pre:
            recommended += 1
        candidates.append(_card(c, pre_ticked=pre,
                                shown_by_default=pre or c.canonical_id in floor_ids))

    shown = sum(1 for c in candidates if c["shown_by_default"])
    # ONE CUT, IN RANK ORDER, NO EXEMPTIONS. `candidates` is already ranked, so a prefix is the
    # top N — and a prefix is the only cut that cannot disagree with the ranking it came from.
    # `shown_by_default` still flags the default view among what is returned; it no longer decides
    # what is returned, which is the whole of the 2026-09-14 fix (see `SCREEN_LIMIT`).
    returned = candidates[:max(0, min(limit, SCREEN_LIMIT_MAX))]
    out = {
        # WITH `source=` THIS IS THE MATCHED TOTAL, not the store's. `source` rides beside it so a
        # reader can never mistake a filtered count for the whole roster.
        "total_candidates": len(candidates),
        "recommended_count": recommended,          # pre-ticked (corroborated persons)
        "shown_by_default_count": shown,           # the ≥floor default view
        "floor": floor,
        "classify": classify,                      # {ran, classified, …} — ran=False = degrade-open
        "candidates": returned,
        "note": ("Pre-ticked = people you've corroborated (≥2 distinct signals) — confirm to make "
                 "them Oracles. Others are shown unchecked, in rank order; each card carries its "
                 "`kind` (person/org/media/project/aggregator), so say what a non-person is rather "
                 "than skipping it. Authors of papers the user saved are never pre-ticked and "
                 "carry no `kind` — describe them from their `reflected` line instead. To add "
                 "someone not listed, pass their handle/URL to "
                 "oracle(action='confirm', add_handles=[...])."),
    }
    if source:
        out["source"] = source
        out["source_note"] = (f"Filtered to people who reach this user through {source}. "
                              f"Ranks and pre-ticks are unchanged — this narrows WHO is listed, "
                              f"not how anyone scored. Omit `source` for the whole roster.")
    if limit > SCREEN_LIMIT_MAX:
        # Said plainly rather than silently honored. The ask behind an oversized `limit` is almost
        # always "show me everyone on platform X" — which `source=` answers without the payload.
        out["limit_clamped"] = {
            "asked": limit, "applied": SCREEN_LIMIT_MAX,
            "note": (f"`limit={limit}` was clamped to {SCREEN_LIMIT_MAX}. The full list of a real "
                     f"store runs past the host's token limit, which is what this bound exists to "
                     f"prevent. To reach a specific group, pass `source=` (one of this store's "
                     f"platforms) or `oracle(action='candidates', query='...')` to search by what "
                     f"they write."),
        }
    if len(returned) < len(candidates):
        # Reported only when it happened. Printed as `omitted: 0` on every ordinary screen it
        # would train the reader to skip the field, and this is the one field that must be read:
        # it is the difference between a bounded list and a list that quietly lost people.
        cut = len(candidates) - len(returned)
        out["omitted"] = {
            "count": cut,
            "note": (f"{cut} more candidates rank below the ones here. They are not gone — raise "
                     f"`limit` to reach further down, a page at a time. Do NOT raise it past ~80: "
                     f"the full list of a real store runs past the token limit, which is what "
                     f"this bound exists to prevent. To see a whole platform's people, pass "
                     f"`source=` ({', '.join(known_platforms(conn)) or 'none yet'}). To find "
                     f"someone specific, `oracle(action='candidates', query='...')` searches by "
                     f"what they write. "
                     f"Mention this only if the user asks whether that is everyone."),
        }
    return out
