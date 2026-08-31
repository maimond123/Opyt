"""
pipeline/kb/reader_core.py — the shared machinery behind every Frontier reading job.

Frontier stage 1 turns the KB into standing queries by READING it. The slice it reads is a seeded
topical region (`sitting_reader.py`) — one conversation, in publication order, complete.

Split out of `sitting_reader.py` when a second reader (`bookmark_reader.py`, since deleted)
existed alongside it: what differs between reading jobs is the slice and the prompt, and
everything downstream of "here is some text, go read it" belongs in one place regardless of
which reading job calls it.

What this owns:
  • the two transports — OpenRouter via `llm_client`, or a headless `claude -p` on a subscription
  • the output contract — parse a possibly-fenced JSON body, then validate it hard
  • provenance repair — cited atom_ids checked against the atoms actually shown to the model

What it deliberately does NOT own: which atoms to read, what to say to the model, when a run is
due, and what "done" means. Those are the jobs' own decisions, and folding them in here would make
the second reader a set of flags on the first instead of its own module.

Validation lives here, not at the decode layer, and that is a real distinction: JSON mode only ever
bought valid JSON, never the right JSON. Every drop is returned as a note so a degrading model
shows up in the run log instead of quietly emitting three usable queries out of twenty.
"""

from __future__ import annotations

import json
import os
import subprocess

# ONE normalizer, imported rather than re-implemented. Verdict matching and row identity must agree
# exactly — two copies that drift by a character would send a verdict to no row at all, silently,
# and the query would look un-verdicted forever. `frontier_queries` imports nothing from here, so
# there is no cycle.
from .frontier_queries import normalize

# The role both readers call under. One role, because the JOB is the same shape — a single long
# read producing standing decisions — and splitting it would mean maintaining two model choices
# that should always move together.
ROLE = "frontier_reader"

# ── Which engine reads the bookmarks ────────────────────────────────────────────
# Two transports, one job. `api` calls OpenRouter through `llm_client`; `claude-cli` shells out to
# a headless `claude -p`, which bills a Claude subscription instead of metered credits.
#
# The shipped default is `api`, and that is an invariant, not a preference. CLAUDE.md requires the
# core to run on ANY MCP client, with Claude Code-specific shell opt-in and never load-bearing —
# a generator that needs the `claude` binary installed and authenticated would silently give
# Cursor/Windsurf/Desktop users no Frontier at all. The CLI path is therefore something a machine
# opts into, via `$OPYT_FRONTIER_BACKEND` or `frontier.backend` in settings.yaml.
BACKEND_API = "api"
BACKEND_CLI = "claude-cli"
CLI_TIMEOUT_S = 1800          # a ~200K-token read runs ~1-3 min; the ceiling is for a wedged child


def frontier_setting(key: str, env_var: str, default: str) -> str:
    """`$ENV` → `frontier.<key>` in settings.yaml → `default`.

    Env first so a one-off run can override without editing config; settings second because the
    detached spawn inherits only the MCP server's environment, not the shell's exports.
    """
    env = os.environ.get(env_var)
    if env:
        return env.strip()
    try:
        from pipeline.ingestion.utils import load_yaml_config
        return str((load_yaml_config().get("frontier") or {}).get(key) or default)
    except Exception:
        return default          # fail-safe: an unreadable config picks the portable default


def resolve_backend() -> str:
    return frontier_setting("backend", "OPYT_FRONTIER_BACKEND", BACKEND_API)


def resolve_cli_model() -> str:
    """Which model the `claude -p` transport reads with.

    These are Claude Code CLI aliases (`sonnet` / `opus` / `haiku`), NOT OpenRouter slugs — this
    transport never touches OpenRouter and has nothing to register in `model_routing.py`.

    Ships as `sonnet` to mirror the tier the `frontier_reader` role uses on the API path.
    """
    return frontier_setting("cli_model", "OPYT_FRONTIER_CLI_MODEL", "sonnet")

# ── Output validation ───────────────────────────────────────────────────────────
# Two different numbers: MAX_NEW_QUERIES bounds how many NEW threads one run may open;
# MAX_QUERIES is the hard clamp on what a single response may add. Conflating them (v1's bug)
# forced new threads to displace survivors at the cap; growth is now decoupled from survival.
# There is no floor — a run opening few new
# threads is a normal week, not a defect.
MAX_NEW_QUERIES = 5
MAX_QUERIES = 25
# The routable vocabulary — the ONLY home for it. A reader's prompt must interpolate this set
# rather than restate it (`sitting_reader._SYSTEM` does), because a name spelled in a prompt but
# absent here is silently dropped by the filter below, and the reader is never told why.
# A name here with no adapter in `frontier_sources.adapters()` is not a bug: the loop records it
# as `no_adapter` and counts it, which is how an unbuilt source stays visible instead of silent.
VALID_SOURCES = frozenset({
    "arxiv", "github", "huggingface", "hackernews", "semantic_scholar",
    "biorxiv", "pubmed", "clinicaltrials", "sec_edgar", "openalex",
})

# The `claims` lens's own bounds (Job N), matching the settled prompt's own "8 to 15 claims" rule.
# NOTE-ONLY on both ends, same as MAX_NEW_QUERIES — a thin region genuinely supports fewer than 8
# checkable claims sometimes, and dropping them would throw away real material over a budget number.
MIN_CLAIMS = 8
MAX_CLAIMS = 15

# ── Response parsing + validation ───────────────────────────────────────────────
def parse_response(text: str) -> dict | None:
    """LLM body → object, tolerant of a fenced or prefixed body. None when unparseable."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1] or t)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


# ── Transport B: a headless `claude -p` on the subscription ─────────────────────
class _CliResponse:
    """Duck-types the fields of `llm_client.LLMResponse` that `_run` reads, so the two transports
    are interchangeable and nothing downstream branches on which one ran."""
    __slots__ = ("text", "model", "input_tokens", "output_tokens", "cost_usd", "raw")

    def __init__(self, text, model, input_tokens, output_tokens, cost_usd, raw):
        self.text, self.model = text, model
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.cost_usd, self.raw = cost_usd, raw


def cli_preflight() -> str | None:
    """None when a headless `claude` is usable, else a human-readable reason (degrade-open)."""
    import shutil
    # `shutil.which`, never a hardcoded path: the binary moves between Homebrew prefixes and npm
    # installs. A shell alias cannot be used here and that is a feature — David's interactive
    # `claude` is aliased to run `brew upgrade` first, which swallows the piped stdin this needs.
    # `subprocess` ignores aliases, so it always gets the real executable.
    return None if shutil.which("claude") else "the `claude` CLI is not on PATH"


def _dominant_model(env: dict) -> str:
    """The model that consumed the most tokens in a `claude -p` run, i.e. the one that did the
    reading — auxiliary models show up in the same `modelUsage` map."""
    mu = env.get("modelUsage") or {}
    if not mu:
        return "?"
    def total(v):
        v = v or {}
        return sum(int(v.get(k) or 0) for k in
                   ("inputTokens", "outputTokens", "cacheCreationInputTokens",
                    "cacheReadInputTokens"))
    return max(mu, key=lambda k: total(mu[k]))


def _cli_failure(proc) -> str:
    """The part of a failed `claude -p` that says WHY.

    On a non-zero exit the CLI still prints its JSON envelope to stdout, and it puts the message in
    `result` — AFTER a `usage` block of zeros. Truncating the raw envelope therefore reports the
    zeros and drops the reason. Measured 2026-08-24: the first sitting read on the live store
    failed and its whole recorded reason was `{"is_error":true,…"output_tokens":0,…"cach` — 400
    characters that say only that nothing happened. Pull the field out instead of widening the
    window, so a longer envelope cannot push the reason back out of range."""
    try:
        env = json.loads(proc.stdout or "")
    except (ValueError, TypeError):
        env = None
    if isinstance(env, dict):
        msg = str(env.get("result") or env.get("error") or "").strip()
        if msg:
            return (f"{msg[:400]} [subtype={env.get('subtype')} "
                    f"stop_reason={env.get('stop_reason')}]")
    return (proc.stderr or proc.stdout or "").strip()[:400]


def call_claude_cli(system: str, user: str, *, model: str | None = None) -> _CliResponse:
    """One bookmark read through a headless `claude -p`. Raises on any failure.

    The contract is "as clean as an API call": a fresh process per run (no history), `cwd` set to
    an empty temp dir (running from the repo leaks CLAUDE.md/project context into the response),
    `--system-prompt` replaces rather than
    appends, no tools/MCP servers, and the window arrives on stdin (~1MB, past any argv limit).

    Deliberately NOT routed through `llm_client`: different transport, different billing surface —
    folding subscription usage into OpenRouter spend accounting would corrupt the paid-sweep caps.
    """
    import shutil
    import tempfile

    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("the `claude` CLI is not on PATH")
    cmd = [exe, "-p",
           "--model", model or resolve_cli_model(),
           "--system-prompt", system,
           "--output-format", "json",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--allowed-tools", ""]
    with tempfile.TemporaryDirectory(prefix="frontier-reader-") as neutral:
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                              timeout=CLI_TIMEOUT_S, cwd=neutral)
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}: {_cli_failure(proc)}")
    try:
        env = json.loads(proc.stdout)
    except (ValueError, TypeError):
        raise RuntimeError(f"claude -p returned non-JSON: {(proc.stdout or '')[:300]}") from None
    if env.get("is_error") or env.get("subtype") != "success":
        raise RuntimeError(f"claude -p reported failure: "
                           f"{env.get('subtype')} {str(env.get('result'))[:300]}")
    usage = env.get("usage") or {}
    return _CliResponse(
        text=str(env.get("result") or ""),
        # The model that did the WORK, not whichever key the dict yielded first — Claude Code bills
        # auxiliary steps to a small model alongside the one that actually read the window.
        model=f"claude-cli:{_dominant_model(env)}",
        # Cache-creation tokens ARE the prompt on this transport — the window is sent once and
        # cached, so counting only `input_tokens` would report ~2 tokens for a 200K-token read.
        input_tokens=int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        # What it WOULD have cost on metered API pricing; no OpenRouter credit is consumed, but
        # subscription quota is finite too, so single-flight keys on this figure.
        cost_usd=float(env.get("total_cost_usd") or 0.0),
        raw=env)


def finish_reason(resp) -> str | None:
    """The provider's stop reason, or None when the response shape does not carry one. Purely
    diagnostic — never load-bearing, since a test fake has no `raw`."""
    try:
        return (resp.raw or {}).get("choices", [{}])[0].get("finish_reason")
    except (AttributeError, IndexError, TypeError):
        return None


def _resolve_atom_ids(cited: list[str], known: set[str] | None) -> tuple[list[str], int]:
    """Map cited ids onto REAL `atoms.atom_id` values; return `(resolved, n_repaired)`.

    Provenance is the one field that must be machine-verifiable, not merely plausible — an
    `atom_id` that joins to nothing is a citation to nowhere, and it fails silently later rather
    than loudly now. So it is checked against the exact window the model was just shown.

    The repair exists because the drift is real and one-directional: the model reliably strips
    the `x:` source prefix from a cited id and returns the bare number. Prefixes are unambiguous
    within a window, so a unique suffix match recovers the true id; an ambiguous or unmatched
    citation is dropped rather than guessed at.
    """
    if known is None:
        return cited, 0
    by_suffix: dict[str, list[str]] = {}
    for k in known:
        by_suffix.setdefault(k.split(":")[-1], []).append(k)
    out, repaired = [], 0
    for c in cited:
        if c in known:
            out.append(c)
            continue
        hits = by_suffix.get(c.split(":")[-1], [])
        if len(hits) == 1:
            out.append(hits[0])
            repaired += 1
    return out, repaired


# ── Did the model read the middle? ──────────────────────────────────────────────
# Catches lost-in-the-middle: a sitting is rendered chronologically, so the middle of the context
# window IS the middle of the time period, and a model that attends poorly there produces reads
# weighted toward the oldest/newest material with no visible error. Runs on every read (not just
# a bake-off) because OpenRouter can route a model to a worse provider between runs, and that
# degradation looks identical to a good read otherwise

# Below this many resolvable citations, ANY claim about shape is noise — five citations land
# wherever they land. The histogram is still returned; only the verdict is withheld.
COVERAGE_MIN_CITATIONS = 8

# How far below its OWN expectation a spread has to fall before it is worth saying. Deliberately
# loose so a rarely-firing metric doesn't get ignored. Measured against 3 bake-off runs.
# n=3, a starting point, not a calibration.
COVERAGE_SPREAD_RATIO = 0.7


def positional_coverage(order: list[str], cited, *, undated=(), buckets: int = 10) -> dict:
    """Where in the rendered document the model's citations actually fell.

    `order` is the chronological sequence the model was shown (must match
    `sitting_render.chronological_order` exactly, or positions mis-attribute). `cited` is the
    resolved atom ids from the response — pass them AFTER `_resolve_atom_ids`, or a hallucinated
    id becomes a phantom position. Undated atoms are excluded, not bucketed, so they don't pile
    into the last decile and fake a recency-bias finding.

    Reads as: `buckets` roughly flat means the model read the whole thing · weight at both ends
    means lost-in-the-middle · weight at the tail means recency bias.

    Fail-safe: an empty region, no citations, or fewer atoms than buckets all return a well-formed
    report rather than raising, since this is an observability signal riding along a paid read.
    """
    undated = set(undated)
    dated = [a for a in order if a not in undated]
    n = len(dated)
    out = {"atoms": n, "buckets": [], "covered": 0, "covered_expected": None, "cited": 0,
           "undated_cited": sum(1 for c in set(cited) if c in undated),
           "middle_share": None, "note": None}
    if not n:
        return out

    # Never more buckets than atoms: with 5 atoms and 10 deciles, five deciles are empty by
    # construction and `covered` would report a degraded read of a perfectly-read region.
    b = max(1, min(buckets, n))
    pos = {a: i for i, a in enumerate(dated)}
    counts = [0] * b
    hits = 0
    for c in set(cited):
        i = pos.get(c)
        if i is None:
            continue
        counts[min(b - 1, i * b // n)] += 1
        hits += 1
    out["buckets"], out["covered"], out["cited"] = counts, sum(1 for c in counts if c), hits
    if not hits:
        out["note"] = "no citations resolved to a dated atom — coverage is unknown, not uniform"
        return out

    # How many sections *should* be touched, given only how many citations there are. Raw `covered`
    # conflates citing sparsely with reading sparsely; without this correction a terse-but-thorough
    # lens reads as lost-in-the-middle.
    out["covered_expected"] = round(b * (1 - (1 - 1 / b) ** hits), 2)

    lo, hi = b // 4, (3 * b) // 4          # the middle half of the document
    out["middle_share"] = round(sum(counts[lo:hi]) / hits, 3)
    if hits < COVERAGE_MIN_CITATIONS:
        out["note"] = f"only {hits} citations — too few to judge shape"
    elif out["middle_share"] < 0.25:
        # Uniform citation puts ~half the weight in the middle half. Half of that is the point
        # where the two ends are doing the work.
        out["note"] = (f"only {out['middle_share']:.0%} of citations fell in the middle half of "
                       f"the document (uniform is ~50%) — the model may be skimming the middle")
    elif out["covered"] < COVERAGE_SPREAD_RATIO * out["covered_expected"]:
        want = out["covered_expected"]
        out["note"] = (f"citations touched {out['covered']} of {b} sections where {want} would be "
                       f"expected from {hits} citations spread evenly — they are clustering")
    return out


def _validate_verdicts(raw, shown: list[str] | None, known_atom_ids: set[str] | None
                       ) -> tuple[list[dict], list[str], int]:
    """`(verdicts, notes, n_repaired)` — one decision per query the reader was actually shown.

    An ABSENT `verdicts` key is silence, not an error — the first-ever run has nothing to judge,
    and some prompts don't ask for verdicts. Matching is on `normalize(text)`, the store's own
    identity, so an echo that drifts in case/spacing still lands on its row; the canonical shown
    text is returned rather than the model's echo. Anything unmatched, or neither `keep` nor
    `drop`, is dropped rather than guessed at.
    """
    notes: list[str] = []
    if raw is None:
        return [], notes, 0
    if not isinstance(raw, list):
        return [], ["'verdicts' present but not a list"], 0
    by_norm = {normalize(t): t for t in (shown or [])}

    out: list[dict] = []
    seen: set[str] = set()
    repaired_total = 0
    for i, v in enumerate(raw):
        if not isinstance(v, dict):
            notes.append(f"verdict[{i}]: not an object")
            continue
        norm = normalize(str(v.get("text") or ""))
        if norm not in by_norm:
            notes.append(f"verdict[{i}] {str(v.get('text'))[:60]!r}: not a query that was shown")
            continue
        call = str(v.get("verdict") or "").strip().lower()
        if call not in {"keep", "drop"}:
            notes.append(f"verdict[{i}] {by_norm[norm]!r}: unreadable verdict {call!r} — "
                         f"treated as no verdict, so the query is untouched")
            continue
        if norm in seen:
            notes.append(f"verdict[{i}] {by_norm[norm]!r}: duplicate verdict, ignored")
            continue
        seen.add(norm)
        ids = v.get("atom_ids")
        ids = [str(a).strip() for a in ids if str(a).strip()] if isinstance(ids, list) else []
        ids, repaired = _resolve_atom_ids(ids, known_atom_ids)
        repaired_total += repaired
        if call == "keep" and not ids:
            # Honoured, not punished. A keep with no usable citation still keeps the query alive;
            # only its provenance goes stale, and `apply_verdicts` leaves the old ids in place so
            # the staleness is visible rather than silently refreshed.
            notes.append(f"verdict[{i}] {by_norm[norm]!r}: kept but uncited — "
                         f"provenance left as it was")
        out.append({"text": by_norm[norm], "verdict": call,
                    "reason": str(v.get("reason") or "").strip() or None,
                    "atom_ids": ids})

    missing = len(by_norm) - len(seen)
    if by_norm and missing > 0:
        # Not an error — a query with no verdict is untouched by design. It is recorded because a
        # reader that stops verdicting is a reader whose whole survival signal has gone quiet, and
        # that must be visible in the run row rather than inferred later from frozen counters.
        notes.append(f"{missing} of {len(by_norm)} shown queries got no verdict (left untouched)")
    return out, notes, repaired_total


def validate(obj: dict, *, known_atom_ids: set[str] | None = None, shown: list[str] | None = None
             ) -> tuple[str, list[dict], list[dict], list[str]]:
    """`(consensus, queries, verdicts, notes)` — well-formed new queries, plus the decisions the
    reader rendered on the queries it was shown.

    Shape is enforced HERE rather than at the decode layer. The role deliberately does not set
    `response_format` (Anthropic + OpenRouter's `require_parameters` cannot route it), but that
    changes nothing about this function's job: JSON mode only ever bought valid JSON, never the
    right JSON. Every drop is returned as a note so a degrading model shows up in the run log
    instead of quietly emitting three usable queries out of twenty.

    Pass `known_atom_ids` to verify provenance against the window that was actually read, and
    `shown` (the standing query texts the prompt listed) to make verdicts meaningful.
    """
    notes: list[str] = []
    consensus = str(obj.get("consensus") or "").strip()
    verdicts, v_notes, repaired_total = _validate_verdicts(
        obj.get("verdicts"), shown, known_atom_ids)
    notes.extend(v_notes)
    raw = obj.get("queries")
    if not isinstance(raw, list):
        # The verdicts still stand — a malformed NEW-QUERIES section says nothing about the
        # decisions already parsed, and discarding them would silently freeze every counter.
        if repaired_total:
            notes.append(f"repaired {repaired_total} atom_id citations to their full ids")
        return consensus, [], verdicts, notes + ["'queries' missing or not a list"]

    out: list[dict] = []
    for i, q in enumerate(raw):
        if not isinstance(q, dict):
            notes.append(f"query[{i}]: not an object")
            continue
        text = str(q.get("text") or "").strip()
        if not text:
            notes.append(f"query[{i}]: empty text")
            continue
        srcs = q.get("target_sources")
        srcs = [s for s in srcs if s in VALID_SOURCES] if isinstance(srcs, list) else []
        if not srcs:
            # No routable source means stage 2 could never execute it. Dropping beats storing a
            # query that silently never runs.
            notes.append(f"query[{i}] {text!r}: no valid target_sources")
            continue
        ids = q.get("atom_ids")
        ids = [str(a).strip() for a in ids if str(a).strip()] if isinstance(ids, list) else []
        ids, repaired = _resolve_atom_ids(ids, known_atom_ids)
        repaired_total += repaired
        if not ids:
            # Provenance is the point. An uncited query — or one whose every citation fails to
            # resolve against the window — cannot be explained or audited later.
            notes.append(f"query[{i}] {text!r}: no resolvable atom_ids")
            continue
        out.append({"text": text, "target_sources": srcs,
                    "rationale": str(q.get("rationale") or "").strip() or None,
                    "atom_ids": ids})

    if repaired_total:
        # Aggregated, not one note per citation — 25 queries drifting the same way is ONE fact
        # about the model, and 60 identical lines would bury the real drops next to it.
        notes.append(f"repaired {repaired_total} atom_id citations to their full ids")

    # A query the reader BOTH kept and re-emitted as new is one sighting, not two. The old prompt
    # trained exactly this behaviour for two months, so it is the likeliest way a run inflates a
    # counter: the verdict and the upsert would each bump `emit_count` for the same query in the
    # same run. The verdict wins, because it carries the decision.
    verdicted = {normalize(v["text"]) for v in verdicts}
    collided = [q for q in out if normalize(q["text"]) in verdicted]
    if collided:
        notes.append(f"{len(collided)} 'new' queries were already verdicted this run "
                     f"(counted once, as verdicts)")
        out = [q for q in out if normalize(q["text"]) not in verdicted]

    if len(out) > MAX_NEW_QUERIES:
        # Over budget is worth SAYING but not worth enforcing at this line: an extra good thread
        # costs one standing query, while dropping it costs the thread. The hard clamp below is
        # the runaway guard.
        notes.append(f"{len(out)} new queries (budget {MAX_NEW_QUERIES})")
    if len(out) > MAX_QUERIES:
        notes.append(f"clamped {len(out)} queries to {MAX_QUERIES}")
        out = out[:MAX_QUERIES]
    return consensus, out, verdicts, notes


def validate_claims(obj: dict, *, known_atom_ids: set[str] | None = None
                    ) -> tuple[list[dict], list[str]]:
    """`(claims, notes)` — well-formed claims with resolved provenance. Job N's `validate`.

    Same discipline as `validate`: shape is enforced here, not at the decode layer, and every drop
    becomes a note rather than a silent hole. `falsified_by` is REQUIRED — a claim with nothing to
    contradict is not the falsifiable claim the prompt asked for (see
    docs/plans/2026-08-16-sitting-rail-model-bakeoff.md). `atom_ids` goes through the same
    `_resolve_atom_ids` repair as a query's citations.
    """
    notes: list[str] = []
    raw = obj.get("claims")
    if not isinstance(raw, list):
        return [], ["'claims' missing or not a list"]

    out: list[dict] = []
    repaired_total = 0
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            notes.append(f"claim[{i}]: not an object")
            continue
        text = str(c.get("claim") or "").strip()
        if not text:
            notes.append(f"claim[{i}]: empty claim")
            continue
        falsified_by = str(c.get("falsified_by") or "").strip()
        if not falsified_by:
            notes.append(f"claim[{i}] {text[:60]!r}: no falsified_by — dropped")
            continue
        ids = c.get("atom_ids")
        ids = [str(a).strip() for a in ids if str(a).strip()] if isinstance(ids, list) else []
        ids, repaired = _resolve_atom_ids(ids, known_atom_ids)
        repaired_total += repaired
        if not ids:
            # Provenance is the point, same as a query: a claim whose every citation fails to
            # resolve against the window it was read from cannot be checked or attributed.
            notes.append(f"claim[{i}] {text[:60]!r}: no resolvable atom_ids — dropped")
            continue
        out.append({"claim": text, "falsified_by": falsified_by, "atom_ids": ids})

    if repaired_total:
        notes.append(f"repaired {repaired_total} atom_id citations to their full ids")
    if len(out) > MAX_CLAIMS:
        notes.append(f"clamped {len(out)} claims to {MAX_CLAIMS}")
        out = out[:MAX_CLAIMS]
    elif 0 < len(out) < MIN_CLAIMS:
        notes.append(f"only {len(out)} claims survived validation (asked for "
                     f"{MIN_CLAIMS}-{MAX_CLAIMS})")
    return out, notes


# ── One door for both transports ────────────────────────────────────────────────
def preflight(backend: str) -> str | None:
    """None when `backend` can be called, else a human-readable reason.

    DEGRADE-OPEN. A missing key, an undeclared role, or an absent `claude` binary is not an error
    to raise — it is a reason to write nothing and say so. The caller records a failed run with
    this string and leaves the existing query set exactly as it found it.
    """
    if backend == BACKEND_CLI:
        return cli_preflight()
    if backend != BACKEND_API:
        return f"unknown backend {backend!r} — expected {BACKEND_API!r} or {BACKEND_CLI!r}"
    try:
        from pipeline import llm_client
    except Exception as e:
        return f"llm_client import failed: {e}"
    try:
        return llm_client.preflight(ROLE)
    except Exception as e:
        return f"role {ROLE!r} unavailable: {e}"


def call(backend: str, system: str, user: str):
    """One completion on `backend`. Raises on failure; the caller decides what a failure costs.

    Returns something with `.text/.model/.input_tokens/.output_tokens/.cost_usd/.raw` either way,
    so nothing downstream branches on which transport ran.
    """
    if backend == BACKEND_CLI:
        return call_claude_cli(system, user)
    from pipeline import llm_client
    return llm_client.call(ROLE, system=system, user=user)
