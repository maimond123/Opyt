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
  • Never HIDE and never REORDER by kind — the payload is in rank order, always. The kind label
    reads name + bio only (often a name alone), which is the weakest evidence in the system;
    letting it move someone out of the default view is the closest thing to hiding a real person.
    So the label's ONLY consequence is the pre-tick — reversing the demotion + persons-only floor
    this file shipped with (David, 2026-08-24, when full classification made them live for ~1000
    people instead of 27). Every card still carries its `kind`, so the host can say what it is.
  • PRE-TICK only corroborated persons — a pre-check is us vouching, and a false-positive Oracle
    is expensive (Stage-5 deep-ingest + becomes a trust root).
  • ENDORSEMENT FIRST — a person the user endorsed outranks anyone they only ever read. See
    `Candidate.sort_key`, which records the 2026-08-23 reversal this replaced.
  • `count` is a SOFT tiebreak only, third behind endorsement and distinct-signal count. Never a
    count-weighted score: a weighted model has to defend a ratio, a lexicographic one only has to
    defend a category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pipeline.timeparse import parse_ts

from . import schema

# The person-endorsement signals — the PRIMARY rank tier: a follow/list/subscribe is the user
# vouching for the PERSON directly; a save/like vouches for a piece of CONTENT (the author is
# inferred). Subscribe and list stay grouped with follow because all three are person-level acts,
# and 23 people in the live store carry one WITHOUT a follow, so the grouping is load-bearing.
_ENDORSEMENT = frozenset({"follow", "list", "subscribe"})

# The 5-way kind vocabulary (Fork 3). Only 'person' is ACTED on (pre-tick eligibility); the other
# four are reported-but-inert — they name the card, they never move or hide it.
_KINDS = ("person", "org", "media", "project", "aggregator")

# Batch size for the classifier. The `entity_classify` role caps OUTPUT at 1024 tokens as shipped,
# and a truncated body fails `json.loads` outright — so an oversized batch loses EVERY label in it,
# not just the tail. 100 fits under 1024 even at the worst case (every verdict the longest kind,
# pretty-printed). 150 does not. Arithmetic: docs/plans/2026-08-24-f5-classify-every-candidate-build.md
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


# The one signal the user actively maintains (can revoke), so it's the only one whose absence is
# meaningful; append-only signals (like/save/subscribe) never veto it. `list` is excluded too —
# too few rows to be a habit yet.
MAINTAINED_SIGNAL = "follow"


def _retired_ids(conn) -> set[str]:
    """Entity ids whose MAINTAINED signal a healthy full walk failed to re-confirm.

    Two gates, and both must pass before absence counts as evidence. The collector's last walk has
    to be trustworthy (`curation_state.walk_is_trustworthy` — an `ok` run whose `found` did not
    collapse), and the signal's own `last_confirmed_at` has to predate that walk.

    Fail-safe is the empty set, and every failure path returns it: no clock, an unreadable
    clock, an untrustworthy walk, an unparseable stamp. Retiring nobody costs a cycle; retiring
    someone a broken walk failed to see costs a signal no later run brings back."""
    try:
        from . import curation_state
        from .ingest_curation import SPEC_BY_COLLECTOR
        spec = next(s for s in SPEC_BY_COLLECTOR.values() if s.signal_type == MAINTAINED_SIGNAL)
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
    except Exception:
        return set()


def rank_candidates(conn, *, include_retired: bool = False) -> list[Candidate]:
    """Group every curation signal by canonical_id and rank the resulting people structurally
    (Fork 1). Candidate universe = SIGNAL-BEARING canonical entities only (an atom-author with no
    curation signal is corpus, not a candidate). Returns the ordered list; empty when no signals.

    RETIRED people are dropped by default — see `MAINTAINED_SIGNAL`. `include_retired=True` returns
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
                if any(r["signal_type"] == MAINTAINED_SIGNAL and r["entity_id"] == m["entity_id"]
                       for r in grp)) and any(r["signal_type"] == MAINTAINED_SIGNAL for r in grp),
        ))

    cands.sort(key=Candidate.sort_key)
    return cands if include_retired else [c for c in cands if not c.retired]


def reflect(cand: Candidate) -> str:
    """Reflect the user's OWN signals back as a short human phrase — "you follow · subscribe ·
    bookmarked 12×". DEGRADES honestly: 'subscribe (paid)' only when is_paid is known True; a
    None/unknown is_paid falls to a plain 'subscribe' (the subscriber-lists endpoint omits paid)."""
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
            parts.append(f"bookmarked {c}×" if pf == "x" else f"saved {c} post(s)")
        elif st == "like":
            parts.append(f"liked {c} of their posts")
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


def classify_kinds(conn, candidates: list[Candidate], *, role: str = "entity_classify") -> dict:
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

    pending = [c for c in candidates if c.kind is None]
    if not pending:
        return {"ran": True, "classified": 0, "note": "every candidate already classified"}

    try:
        from pipeline import llm_client
    except Exception as e:
        return {"ran": False, "reason": f"llm_client import failed: {e}", "classified": 0}
    # preflight: a missing role or absent key degrades OPEN rather than raising into the screen.
    # A global condition, so it is checked ONCE — a missing key should cost one check, not ten.
    try:
        reason = llm_client.preflight(role)
    except Exception as e:
        reason = f"role {role!r} unavailable: {e}"
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
            resp = llm_client.call(role, system=_CLASSIFY_SYSTEM, user=_classify_prompt(batch))
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
    return out


# ── (c) assembly ─────────────────────────────────────────────────────────────────

def _card(cand: Candidate, *, pre_ticked: bool, shown_by_default: bool) -> dict:
    return {
        "canonical_id": cand.canonical_id,
        "name": cand.name,
        "handle": cand.handle,
        "kind": cand.kind or "unclassified",
        "is_person": cand.is_person,
        "corroborated": cand.corroborated,
        "pre_ticked": pre_ticked,
        "shown_by_default": shown_by_default,
        "distinct_signals": cand.distinct_signals,
        "total_count": cand.total_count,
        "reflected": reflect(cand),
        "signals": cand.signals,
        "identity_links": cand.identity_links,
        "members": cand.members,
    }


def build_screen(conn, *, floor: int = DEFAULT_FLOOR) -> dict:
    """The full SCREEN payload the `oracle` tool hands the host. Ranks, classifies every
    unclassified candidate (degrade-open), then partitions:
      • pre_ticked       = corroborated (distinct≥2) AND person — the default-YES vouch set.
      • shown_by_default = pre_ticked OR within the visibility floor, in RANK order.
      • the rest ride behind "see all", also in rank order.
    Nothing is hidden and nothing is reordered by kind — every signal-bearing candidate is in
    `candidates` in rank order, each flagged with its `kind` so the host can say what it is. See
    the module docstring for why the label stops at the pre-tick."""
    ranked = rank_candidates(conn)
    classify = classify_kinds(conn, ranked)

    # Floor filled in RANK order, kind-blind: an org at rank 3 keeps rank 3 and its place in the
    # default view; it just does not arrive pre-ticked.
    floor_ids = {c.canonical_id for c in ranked[:floor]}

    candidates, recommended = [], 0
    for c in ranked:
        pre = c.corroborated and c.is_person
        if pre:
            recommended += 1
        candidates.append(_card(c, pre_ticked=pre,
                                shown_by_default=pre or c.canonical_id in floor_ids))

    shown = sum(1 for c in candidates if c["shown_by_default"])
    return {
        "total_candidates": len(candidates),
        "recommended_count": recommended,          # pre-ticked (corroborated persons)
        "shown_by_default_count": shown,           # the ≥floor default view
        "floor": floor,
        "classify": classify,                      # {ran, classified, …} — ran=False = degrade-open
        "candidates": candidates,
        "note": ("Pre-ticked = people you've corroborated (≥2 distinct signals) — confirm to make "
                 "them Oracles. Others are shown unchecked, in rank order; each card carries its "
                 "`kind` (person/org/media/project/aggregator), so say what a non-person is rather "
                 "than skipping it. To add someone not listed, pass their handle/URL to "
                 "oracle(action='confirm', add_handles=[...])."),
    }
