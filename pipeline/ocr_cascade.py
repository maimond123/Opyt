"""
pipeline/ocr_cascade.py — OCR-first media understanding for footprint ingest.

Moved here from `pipeline/processing/` (the vault producer package, queued for deletion)
because the atom-KB ingesters and `pipeline/kb/vision.py` run every image through it. Nothing
in this file is vault-specific; only its address changed.

Replaces "describe every image with one VLM" (describe_images.describe_image) on the footprint
path. A cheap OCR pass transcribes AND tags the image; the tag routes:

  image → cheap OCR/classify (gemma-3-27b: transcribe ALL text verbatim + a TYPE tag)
    ├─ SCREENSHOT / document-dense  → keep the OCR transcription VERBATIM (lossless)   → SUBSTANCE
    ├─ CHART / diagram (sparse text) → detailed chart VLM (gemini-2.5-flash)            → SUBSTANCE
    └─ PHOTO / NO_TEXT              → no atom text                                      → DECORATIVE

A screenshot-of-text is TRANSCRIBED losslessly, not paraphrased; a chart gets an analytic VLM
read; a selfie/logo/meme carries no atom substance and is marked DECORATIVE. `substance` is the
media half of the artifact-vs-fragment decision: a well-read chart/screenshot is an ARTIFACT, a
photo is a FRAGMENT.
notes behind this shape.

The router uses the model's own TYPE tag; a char-count density check is only a backstop when the
tag is missing or wrong.

Model choice: cheap vision LLMs via a model override on the existing `vision` role, not a new
settings role — a new `ocr` role would KeyError on any settings.yaml written before it existed.

Fail-safe (poison-value rule, mirrors describe_images): ANY failure → return None (skip, do NOT
cache, retry next run), never crash the ingest. Cost is never gated here: `llm_client.call`
records what each hop actually cost, per model, into api_stats.json — this module holds no
price table and computes no spend of its own.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from pipeline import llm_client
from pipeline.concurrency import AdaptiveSemaphore
from pipeline.ingestion.utils import log

# The image-call throttle + telemetry, owned here (moved from `describe_images` along with the
# image seam). ONE process-wide AdaptiveSemaphore caps concurrent image calls (AIMD: climb on
# success, halve on 429) — it must stay a single instance, or two gates would each think they cap
# at 8 while admitting 16 combined.
_VLM_GATE = AdaptiveSemaphore(8, min_permits=2, max_permits=12, increase_after=8)

# Bounded per-image latency telemetry (no unbounded sample list — the MCP server is long-lived).
# The only seam that sees per-image latency; callers wanting per-run figures snapshot/diff these.
_STATS = {"calls": 0, "failures": 0, "seconds": 0.0, "max_seconds": 0.0}
_STATS_LOCK = threading.Lock()          # many producer threads read images at once → guard the RMW


def _is_rate_limited(exc: Exception) -> bool:
    """A failure that is specifically a 429 (rate-limit), not a 402/credit or network error — the
    signal that tells the AIMD gate to back off rather than treating it as a one-off skip.
    llm_client surfaces provider errors as `_BackendError('HTTP 429: ...')`, so we match the code."""
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "rate-limit" in s or "too many requests" in s


def stats_snapshot() -> dict:
    """A copy of the additive counters, for a caller that wants THIS run's delta."""
    with _STATS_LOCK:
        return dict(_STATS)

# Cheap vision LLM on OpenRouter for OCR; a stronger model for chart reads. Both are model
# OVERRIDES on the `vision` role (no new settings key), chosen for provider redundancy — served by
# four independent upstream orgs, not one.
# benchmark/rejection history.
# The slugs live in `pipeline/model_routing.py` — the ONE place a model literal may appear
# (enforced by the `bare-model-slug` guard) — so the availability preflight there is complete.
from pipeline.model_routing import CHART_MODEL, OCR_MODEL  # noqa: E402  (re-exported below)

# NO price table here. `llm_client.call` already records this stage's spend per model — from the
# charge OpenRouter REPORTS, not an estimate — into api_stats.json's `by_model`/`by_rail` buckets,
# and a model override is applied before that record, so a fallback run buckets under the fallback.
# A table here would be a second, less accurate copy of a number the system already has exactly.

# Below this many transcribed chars a 'SCREENSHOT' tag is untrustworthy (a logo with a word or
# two) → treat as a photo; above it, a 'PHOTO' tag on a text-wall → promote to a document.
_DENSE_CHARS = 60

_OCR_PROMPT = (
    "Transcribe ALL text visible in this image verbatim, exactly as written, preserving line "
    "order. Do not describe or summarize — transcribe. Then, on a FINAL separate line, output "
    "exactly one tag:\n"
    "'TYPE: SCREENSHOT' — a screenshot, document, table, or post that is mostly text;\n"
    "'TYPE: CHART' — a chart, graph, plot, diagram, or data visualization;\n"
    "'TYPE: PHOTO' — a photo, logo, meme, or illustration with little or no meaningful text.\n"
    "If there is no readable text at all, transcribe nothing and output only: TYPE: PHOTO")

_CHART_PROMPT = (
    "This image is a chart, graph, or diagram. Extract, concretely: the title; each axis and its "
    "unit; every data series or category; the specific values / ranges / standout points; the "
    "overall trend; and the single most important takeaway. Be quantitative. Do not speculate "
    "beyond what is shown. "
    # The model answers conversationally unless told not to (e.g. "Here's a detailed extraction
    # of information from the provided chart:"); this instruction stops that lead-in at the source.
    "Begin directly with the title. Do not open with a sentence about what you are about to do.")

# Strips a chart read's conversational lead-in sentence (e.g. "Here's a detailed extraction of
# information from the provided chart:"). Bounded/anchored so it only ever matches a short opener,
# never real content that happens to contain a colon. Hygiene only — not a fix for cross-atom
# similarity;
_LEAD_IN = re.compile(r"\A\s*Here(?:'s|’s|\s+is)\s+[^:\n]{0,120}:\s*", re.I)


def _strip_lead_in(text: str) -> str:
    """Drop a chart read's opening "Here's a ...:" sentence. Idempotent; "" stays ""."""
    return _LEAD_IN.sub("", text or "")


@dataclass
class MediaRead:
    """The structured result of reading one image. `substance` is the media half of the
    artifact-vs-fragment call: True for a document/chart (retrievable content), False for a
    decorative photo. `text` is what gets rendered into the atom (searchable)."""
    text: str
    kind: str            # "document" | "chart" | "photo"
    substance: bool

    def __post_init__(self) -> None:
        # Normalize here (not at each call site) so every construction path — fresh read or cache
        # rehydration — gets the lead-in stripped, with no path able to skip it silently.
        self.text = _strip_lead_in(self.text)

    def to_cache(self) -> dict:
        """Persisted-by-URL form (immutable CDN link)."""
        return {"text": self.text, "kind": self.kind, "substance": self.substance}


def from_cache(v) -> "MediaRead | None":
    """Rehydrate a cached read, or None when the entry CANNOT be trusted (e.g. a legacy bare
    string from a different ingester's shape) — the caller must treat None exactly like a cache
    miss: re-read and overwrite.
    at the legacy shape is unsafe."""
    if isinstance(v, dict):
        return MediaRead(v.get("text", ""), v.get("kind", "document"), bool(v.get("substance")))
    return None


def _parse_tag(text: str) -> tuple[str, str]:
    """Split the OCR output into (transcription, TAG). The tag is the last 'TYPE: X' line;
    every non-tag line is the transcription."""
    tag, body = "", []
    for ln in (text or "").splitlines():
        if ln.strip().upper().startswith("TYPE:"):
            tag = ln.strip().split(":", 1)[1].strip().upper()
        else:
            body.append(ln)
    return "\n".join(body).strip(), tag


def _route(transcription: str, tag: str) -> str:
    """document | chart | photo — the model's tag, with a density backstop for a wrong/missing tag."""
    dense = len(transcription) >= _DENSE_CHARS
    if tag == "CHART":
        return "chart"
    if tag == "SCREENSHOT":
        return "document" if dense else "photo"   # a 'screenshot' with ~no text is really a photo
    return "document" if dense else "photo"        # PHOTO / missing: density can still promote


# Set once when the OCR model proves UNROUTABLE, so the rest of the run short-circuits instead of
# paying a doomed round-trip per image. Process-scoped on purpose: the condition is a config fact
# that cannot change mid-run, and the next process re-checks from scratch.
_STAGE_DISABLED: str | None = None

# The OCR model this run actually calls — `resolve_ocr_model`'s pick, resolved ONCE per process
# (the resolution walks `surviving_orgs` per candidate; the OCR path is per-image).
_RESOLVED: str | None = None


def stage_status() -> str | None:
    """None when OCR is live; otherwise WHY it is off — for a run summary to surface. The 2026-08
    outage was expensive precisely because 'no transcripts' looked identical to 'no images'."""
    return _STAGE_DISABLED


def _reset_stage_for_tests() -> None:
    global _STAGE_DISABLED, _RESOLVED
    _STAGE_DISABLED = None
    _RESOLVED = None


def _ocr_model() -> str | None:
    """The model to transcribe with — `OCR_FALLBACKS` walked once per process. None means the
    stage is disabled (`_STAGE_DISABLED` carries why). A substitution is LOGGED at the same
    volume as STAGE DISABLED: a silent swap is the failure class this module exists to prevent.
    Fail-safe: a resolution that throws proceeds with the primary — preflight must never be the
    reason a run cannot start."""
    global _RESOLVED, _STAGE_DISABLED
    if _RESOLVED is not None:
        return _RESOLVED
    try:
        from pipeline.model_routing import resolve_ocr_model
        model, reason = resolve_ocr_model()
    except Exception as e:
        model, reason = OCR_MODEL, f"resolution failed ({e}) — proceeding with the primary"
    if model is None:
        _STAGE_DISABLED = (f"{reason}. Images will have NO transcripts for this run. Edit the "
                           f"OpenRouter deny-list to restore a route — the `oracle` screen's "
                           f"`model_routing` notice shows what survives.")
        log(f"[ocr] STAGE DISABLED — {_STAGE_DISABLED}")
        return None
    if model != OCR_MODEL:
        log(f"[ocr] {reason}")               # reason carries the "FALLBACK ..." wording
    _RESOLVED = model
    return model


def _warn_if_near_ceiling(url: str, out_tokens: int, max_tokens: int) -> None:
    """A response landing near its max_tokens ceiling is a truncation/degeneration RISK, not a
    confirmed one — but silent proximity is exactly how a dense-image transcription bug went
    unnoticed for weeks."""
    if out_tokens >= max_tokens * 0.9:
        log(f"[ocr] {url} used {out_tokens}/{max_tokens} output tokens — possible truncation")


def _record(t0: float, *, failed: bool = False) -> None:
    """Fold one image call's outcome into `_STATS`. Counted on the FIRST (transcribe) hop only —
    that hop happens for every image, so `calls` stays a clean image count; the conditional chart
    hop would make it a call count that no caller could interpret."""
    dt = time.perf_counter() - t0
    with _STATS_LOCK:
        _STATS["seconds"] += dt
        _STATS["max_seconds"] = max(_STATS["max_seconds"], dt)
        if failed:
            _STATS["failures"] += 1


def read_image(image_url: str, *, context: str = "") -> "MediaRead | None":
    """Run the OCR cascade on ONE image. Returns a MediaRead, or None on failure (skip, do NOT
    cache — poison-value rule). `context` (the post text) grounds the chart read."""
    global _STAGE_DISABLED
    if _STAGE_DISABLED:
        return None            # already known impossible — do not re-pay the round-trip
    model = _ocr_model()
    if model is None:
        return None            # every declared candidate is dead — `_ocr_model` already logged why
    with _STATS_LOCK:
        _STATS["calls"] += 1
    t0 = time.perf_counter()
    try:
        with _VLM_GATE:        # process-wide AIMD cap — see the note at the top of this module
            r = llm_client.call(role="vision", system="", user=_OCR_PROMPT,
                                images=[image_url], max_tokens=6000, model=model,
                                frequency_penalty=0.5)   # a dense/repetitive image can loop otherwise
        _VLM_GATE.record_success()      # clean 2xx → probe the limit up a notch (eventually)
    except llm_client.ModelUnroutableError as e:
        # TERMINAL: no provider can serve this model, so every remaining image would fail
        # identically. Disable the stage for the run and say so LOUDLY — the failure mode this
        # replaces was hundreds of silent 404s that also poisoned the shared breaker.
        _STAGE_DISABLED = (f"OCR model {model!r} is unroutable under the active deny-list "
                           f"({e}). Images will have NO transcripts for this run. Edit the "
                           f"OpenRouter deny-list to restore a route — the `oracle` screen's "
                           f"`model_routing` notice shows what survives.")
        log(f"[ocr] STAGE DISABLED — {_STAGE_DISABLED}")
        _record(t0, failed=True)
        return None
    except Exception as e:
        if _is_rate_limited(e):
            _VLM_GATE.decrease()        # 429 → halve the limit (back off, don't just skip)
        log(f"[ocr] transcribe failed {image_url}: {e}")
        _record(t0, failed=True)
        return None
    _record(t0)
    _warn_if_near_ceiling(image_url, r.output_tokens, 6000)

    transcription, tag = _parse_tag(r.text)
    kind = _route(transcription, tag)

    if kind == "chart":
        chart_user = _CHART_PROMPT + (f"\n\nContext from the post:\n{context[:400]}" if context else "")
        try:
            with _VLM_GATE:    # the chart hop is an image call too — same throttle, not a bypass
                c = llm_client.call(role="vision", system="", user=chart_user,
                                    images=[image_url], max_tokens=2000, model=CHART_MODEL)
            _VLM_GATE.record_success()
            _warn_if_near_ceiling(image_url, c.output_tokens, 2000)
        except Exception as e:
            if _is_rate_limited(e):
                _VLM_GATE.decrease()
            # Degrade, don't crash: keep the OCR transcription as document-ish substance.
            log(f"[ocr] chart read failed {image_url} (keeping OCR text): {e}")
            return MediaRead(transcription, "document" if transcription else "photo",
                             bool(transcription))
        return MediaRead(c.text.strip(), "chart", True)

    if kind == "document":
        return MediaRead(transcription, "document", True)

    return MediaRead("", "photo", False)                                    # decorative
