"""
pipeline/chart_read.py — reads charts out of images; everything else is NOT_CHART.

The atom-KB ingesters and `pipeline/kb/vision.py` run every image they read through it. One call
on `CHART_MODEL` either extracts the chart (title, axes, series, values, trend, takeaway) or
answers exactly `NOT_CHART`. Screenshots, documents, tables of numbers, diagrams with no data,
photos and memes keep no text (only chart reads were useful). A `NOT_CHART`
is a verdict and is cached like a chart; a failed or empty reply returns None (skip, no cache
write, retry next run), never a crash.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from opyt_core import config
from pipeline import llm_client, model_routing
from pipeline.concurrency import AdaptiveSemaphore
from pipeline.ingestion.utils import log
# An override on the `vision` role. The slug lives in `pipeline/model_routing.py`, the one place a
# model literal may appear.
from pipeline.model_routing import CHART_MODEL

# One process-wide AIMD gate caps concurrent image calls. It must stay a single instance, or two
# gates would each cap at 8 while admitting 16 combined.
_VLM_GATE = AdaptiveSemaphore(8, min_permits=2, max_permits=12, increase_after=8)

# Bounded per-image counters (the MCP server is long-lived); callers snapshot/diff for per-run figures.
_STATS = {"calls": 0, "failures": 0, "seconds": 0.0, "max_seconds": 0.0,
          "byok_only_skipped": 0}
_STATS_LOCK = threading.Lock()          # many producer threads read images at once → guard the RMW


def _is_rate_limited(exc: Exception) -> bool:
    """A failure that is specifically a 429, the signal for the AIMD gate to back off. Matched on
    the message because llm_client surfaces it as `_BackendError('HTTP 429: ...')`."""
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "rate-limit" in s or "too many requests" in s


def stats_snapshot() -> dict:
    """A copy of the additive counters, for a caller that wants this run's delta."""
    with _STATS_LOCK:
        return dict(_STATS)


_CHART_PROMPT = (
    "If this image is a chart, graph or plot of data, extract, concretely: the title; each axis "
    "and its unit; every data series or category; the specific values / ranges / standout "
    "points; the overall trend; and the single most important takeaway. If it holds several "
    "charts, extract each. Be quantitative. Do not speculate beyond what is shown. "
    # Stops the conversational "Here's a detailed extraction..." lead-in at the source.
    "Begin directly with the title. Do not open with a sentence about what you are about to do.\n"
    "If the image is anything else — a screenshot, a document, a table of numbers, a diagram "
    "with no data, a photo, a meme, a logo — answer exactly NOT_CHART and nothing else.")

NOT_CHART = "NOT_CHART"

# A chart read's conversational lead-in ("Here's a detailed extraction of ...:"). Anchored and
# bounded so it only matches a short opener, never real content containing a colon.
_LEAD_IN = re.compile(r"\A\s*Here(?:'s|’s|\s+is)\s+[^:\n]{0,120}:\s*", re.I)


def _strip_lead_in(text: str) -> str:
    """Drop a chart read's opening "Here's a ...:" sentence. Idempotent; "" stays ""."""
    return _LEAD_IN.sub("", text or "")


@dataclass
class MediaRead:
    """The result of reading one image: a chart's extraction, or `not_chart` with no text. `text`
    is what gets rendered into the atom (searchable)."""
    text: str
    kind: str            # "chart" | "not_chart"

    def __post_init__(self) -> None:
        # Normalized here so every construction path (fresh read or cache) strips the lead-in.
        self.text = _strip_lead_in(self.text)

    @property
    def substance(self) -> bool:
        """The media half of the substance filter: only a chart is retrievable content."""
        return self.kind == "chart"

    def to_cache(self) -> dict:
        """Persisted-by-URL form (immutable CDN link)."""
        return {"text": self.text, "kind": self.kind}


# Cached kinds from before the reader was charts-only. A `document` was an OCR transcription and is
# never replayed as text: it reads as a miss, so the image is read again. A `photo` had no text
# and no chart tag. A `chart` was a chart read and stays one.
_LEGACY_KINDS = {"photo": "not_chart", "document": None}


def from_cache(v) -> "MediaRead | None":
    """Rehydrate a cached read, or None when the entry cannot be trusted (a legacy bare string, a
    legacy transcription); the caller treats None like a cache miss: re-read and overwrite."""
    if not isinstance(v, dict):
        return None
    kind = v.get("kind")
    kind = _LEGACY_KINDS.get(kind, kind)
    if kind not in ("chart", "not_chart"):
        return None
    return MediaRead(v.get("text", "") if kind == "chart" else "", kind)


def verdict(reply: str) -> "MediaRead | None":
    """A model reply → its read: empty is None (a failure, retried), `NOT_CHART` is a cached
    verdict with no text, anything else is the chart's extraction."""
    text = (reply or "").strip()
    if not text:
        return None
    if text.strip("`*. \n").upper() == NOT_CHART:
        return MediaRead("", "not_chart")
    return MediaRead(text, "chart")


# Set once when the chart model proves unroutable, so the rest of the process short-circuits; the
# next process re-checks.
_STAGE_DISABLED: str | None = None

# The model this process calls, once `chart_model_unroutable` has passed it (checked once per process).
_RESOLVED: str | None = None


def stage_status() -> str | None:
    """None when chart reads are live; otherwise why they are off, for a run summary to surface
    (so 'no chart reads' never looks like 'no images')."""
    return _STAGE_DISABLED


def _reset_stage_for_tests() -> None:
    global _STAGE_DISABLED, _RESOLVED, _BYOK_SKIP_LOGGED
    _STAGE_DISABLED = None
    _RESOLVED = None
    _BYOK_SKIP_LOGGED = False


# Log-once flag for the BYOK gate. Only the log line is once per process; the gate itself is
# re-evaluated on every call, because a user who connects their own key mid-session must be read
# on the very next image.
_BYOK_SKIP_LOGGED = False


def _vision_enabled() -> bool:
    """The BYOK user's own off-switch, `vision.enabled` in settings.yaml. A missing or unreadable
    setting reads as enabled."""
    try:
        val = (config.settings().get("vision") or {}).get("enabled")
    except Exception:
        return True
    return True if val is None else bool(val)


def refused() -> str | None:
    """Why an image read may not be spent here now, or None when it may: the `vision` role is
    refused (a free install has no key), or a BYOK user turned vision off. A broken preflight
    fails safe to None."""
    try:
        role = llm_client.preflight("vision")
    except Exception:
        role = None
    if role:
        return f"image reads skipped: {role}; read later if that changes"
    if not _vision_enabled():
        return "image reads are turned off (`vision.enabled: false` in settings.yaml)"
    return None


def _byok_only_skip() -> bool:
    """True when this call must not spend an image read (`refused`). Then `read_image` returns
    None with no cache write, so a later key connect re-reads everything."""
    global _BYOK_SKIP_LOGGED
    reason = refused()
    if reason is None:
        return False
    with _STATS_LOCK:
        _STATS["byok_only_skipped"] += 1
    if not _BYOK_SKIP_LOGGED:
        _BYOK_SKIP_LOGGED = True
        log(f"[charts] {reason}")
    return True


def _chart_model() -> str | None:
    """`CHART_MODEL` once it is known to be served, checked once per process. None means the stage
    is disabled (`_STAGE_DISABLED` carries why); a check that throws proceeds with the model."""
    global _RESOLVED, _STAGE_DISABLED
    if _RESOLVED is not None:
        return _RESOLVED
    try:
        reason = model_routing.chart_model_unroutable()
    except Exception:
        reason = None
    if reason:
        _STAGE_DISABLED = (f"{reason}. Images will have NO chart reads for this run. Edit the "
                           f"OpenRouter deny-list to restore a route — the `oracle` screen's "
                           f"`model_routing` notice shows what survives.")
        log(f"[charts] STAGE DISABLED — {_STAGE_DISABLED}")
        return None
    _RESOLVED = CHART_MODEL
    return _RESOLVED


def _warn_if_near_ceiling(url: str, out_tokens: int, max_tokens: int) -> None:
    """Log a response landing near its max_tokens ceiling: a possible truncation or loop."""
    if out_tokens >= max_tokens * 0.9:
        log(f"[charts] {url[:200]} used {out_tokens}/{max_tokens} output tokens — possible truncation")


def _record(t0: float, *, failed: bool = False) -> None:
    """Fold one image call's outcome into `_STATS`."""
    dt = time.perf_counter() - t0
    with _STATS_LOCK:
        _STATS["seconds"] += dt
        _STATS["max_seconds"] = max(_STATS["max_seconds"], dt)
        if failed:
            _STATS["failures"] += 1


def read_image(image_url: str, *, context: str = "") -> "MediaRead | None":
    """Read one image: a chart, or `NOT_CHART`. Returns a MediaRead, or None on failure (skip, no
    cache write). `context` (the post text) grounds the chart read."""
    global _STAGE_DISABLED
    if _byok_only_skip():
        return None            # not this key's spend — skip, no cache write, re-read on BYOK
    if _STAGE_DISABLED:
        return None            # already known impossible — do not re-pay the round-trip
    model = _chart_model()
    if model is None:
        return None            # the model is unroutable — `_chart_model` already logged why
    with _STATS_LOCK:
        _STATS["calls"] += 1
    user = _CHART_PROMPT + (f"\n\nContext from the post:\n{context}" if context else "")
    t0 = time.perf_counter()
    try:
        with _VLM_GATE:
            r = llm_client.call(role="vision", system="", user=user,
                                images=[image_url], max_tokens=2000, model=model)
        _VLM_GATE.record_success()      # clean 2xx → probe the limit up a notch (eventually)
    except llm_client.ModelUnroutableError as e:
        # Terminal: every remaining image would fail identically, so disable the stage and say so.
        _STAGE_DISABLED = (f"chart model {model!r} is unroutable under the active deny-list "
                           f"({e}). Images will have NO chart reads for this run. Edit the "
                           f"OpenRouter deny-list to restore a route — the `oracle` screen's "
                           f"`model_routing` notice shows what survives.")
        log(f"[charts] STAGE DISABLED — {_STAGE_DISABLED}")
        _record(t0, failed=True)
        return None
    except Exception as e:
        if _is_rate_limited(e):
            _VLM_GATE.decrease()        # 429 → halve the limit (back off, don't just skip)
        log(f"[charts] read failed {image_url[:200]}: {e}")
        _record(t0, failed=True)
        return None
    _warn_if_near_ceiling(image_url, r.output_tokens, 2000)
    mr = verdict(r.text)
    _record(t0, failed=mr is None)
    return mr
