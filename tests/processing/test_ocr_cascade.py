"""ocr_cascade — the OCR-first media reader (transcribe → route by type → chart-VLM), fully offline.

Proves the ROUTING (the model's TYPE tag + a density backstop), that a document skips the chart
call, that a chart triggers the second model, that a photo is decorative (no substance), and the
fail-safe contract (OCR failure → None/skip; chart failure → degrade to the OCR text, never crash).
The live model behavior is proven by the sandbox HTML report, not here.
"""
from __future__ import annotations

from pipeline import ocr_cascade


class _Resp:
    """A minimal LLMResponse stand-in — the cascade reads only .text/.input_tokens/.output_tokens."""
    def __init__(self, text, i=100, o=40):
        self.text, self.input_tokens, self.output_tokens = text, i, o


# ── pure parsing / routing ─────────────────────────────────────────────────────────

def test_parse_tag_splits_transcription_and_tag():
    body, tag = ocr_cascade._parse_tag("line one\nline two\nTYPE: CHART")
    assert body == "line one\nline two" and tag == "CHART"


def test_parse_tag_missing_tag_is_empty():
    body, tag = ocr_cascade._parse_tag("just some text, no tag line")
    assert tag == "" and body == "just some text, no tag line"


def test_route_dense_screenshot_is_document():
    assert ocr_cascade._route("x" * 100, "SCREENSHOT") == "document"


def test_route_sparse_screenshot_is_photo():                 # a 'screenshot' with ~no text = a photo
    assert ocr_cascade._route("hi", "SCREENSHOT") == "photo"


def test_route_chart_tag_wins():
    assert ocr_cascade._route("Q1 Q2", "CHART") == "chart"


def test_route_no_text_is_photo():
    assert ocr_cascade._route("", "PHOTO") == "photo"


def test_route_dense_untagged_promoted_to_document():        # density backstop for a missing tag
    assert ocr_cascade._route("y" * 100, "") == "document"


# ── the cascade (mocked llm_client) ─────────────────────────────────────────────────

def test_document_keeps_verbatim_and_skips_chart_call(monkeypatch):
    calls = []

    def fake(role, **kw):
        calls.append(kw["model"])
        return _Resp("Benchmark table:\n" + "row data " * 20 + "\nTYPE: SCREENSHOT")

    monkeypatch.setattr(ocr_cascade.llm_client, "call", fake)
    mr = ocr_cascade.read_image("http://img")
    assert mr.kind == "document" and mr.substance is True and "Benchmark table" in mr.text
    assert calls == [ocr_cascade.OCR_MODEL]                   # NO chart-model call


def test_chart_triggers_second_model(monkeypatch):
    def fake(role, **kw):
        if kw["model"] == ocr_cascade.OCR_MODEL:
            return _Resp("2024 2025\nTYPE: CHART")
        return _Resp("Title: Revenue by year. Y-axis $M. 2024=10, 2025=25. Up-trend.")

    monkeypatch.setattr(ocr_cascade.llm_client, "call", fake)
    mr = ocr_cascade.read_image("http://img", context="revenue")
    assert mr.kind == "chart" and mr.substance is True
    assert "Revenue by year" in mr.text                       # the chart read replaced the OCR text


def test_photo_is_decorative(monkeypatch):
    monkeypatch.setattr(ocr_cascade.llm_client, "call", lambda role, **kw: _Resp("TYPE: PHOTO"))
    mr = ocr_cascade.read_image("http://img")
    assert mr.kind == "photo" and mr.substance is False and mr.text == ""


def test_ocr_failure_returns_none_not_crash(monkeypatch):    # poison-value rule: skip, don't cache
    def boom(role, **kw):
        raise RuntimeError("402 out of credits")

    monkeypatch.setattr(ocr_cascade.llm_client, "call", boom)
    assert ocr_cascade.read_image("http://img") is None


def test_chart_failure_degrades_to_document(monkeypatch):    # degrade-to-keep, never lose the read
    def fake(role, **kw):
        if kw["model"] == ocr_cascade.OCR_MODEL:
            return _Resp("some axis labels here long enough to be dense\nTYPE: CHART")
        raise RuntimeError("chart model down")

    monkeypatch.setattr(ocr_cascade.llm_client, "call", fake)
    mr = ocr_cascade.read_image("http://img")
    assert mr.kind == "document" and mr.substance is True and "axis labels" in mr.text


# ── truncation-proximity warning (near max_tokens, not a confirmed failure) ────────

def test_warns_when_ocr_output_near_ceiling(monkeypatch):
    monkeypatch.setattr(ocr_cascade.llm_client, "call",
                        lambda role, **kw: _Resp("dense table\nTYPE: SCREENSHOT", o=5600))
    logged = []
    monkeypatch.setattr(ocr_cascade, "log", logged.append)
    ocr_cascade.read_image("http://img")
    assert any("possible truncation" in m for m in logged)


def test_no_warning_when_ocr_output_well_under_ceiling(monkeypatch):
    monkeypatch.setattr(ocr_cascade.llm_client, "call",
                        lambda role, **kw: _Resp("short caption\nTYPE: SCREENSHOT", o=40))
    logged = []
    monkeypatch.setattr(ocr_cascade, "log", logged.append)
    ocr_cascade.read_image("http://img")
    assert not any("possible truncation" in m for m in logged)


def test_warns_when_chart_output_near_ceiling(monkeypatch):
    def fake(role, **kw):
        if kw["model"] == ocr_cascade.OCR_MODEL:
            return _Resp("2024 2025\nTYPE: CHART")
        return _Resp("Title: dense multi-series chart", o=1900)

    monkeypatch.setattr(ocr_cascade.llm_client, "call", fake)
    logged = []
    monkeypatch.setattr(ocr_cascade, "log", logged.append)
    ocr_cascade.read_image("http://img", context="revenue")
    assert any("possible truncation" in m for m in logged)


def test_from_cache_rehydrates_a_cascade_entry():
    mr = ocr_cascade.from_cache({"text": "a table", "kind": "document", "substance": True})
    assert mr.kind == "document" and mr.substance is True


def test_from_cache_refuses_to_guess_at_a_legacy_string():
    """CHANGED 2026-08-02, and the change is the point. This used to return
    `MediaRead(s, "document" if s else "photo", bool(s))` — inferring the substance verdict from
    "is the string non-empty".

    `image_descriptions.json` is ONE key space (bare URL) shared by five ingesters, and it held two
    value shapes: this cascade's dict, and `describe_image`'s bare prose gloss. `substance` is not
    cosmetic — it flows into `_has_substantive_media` → `_keep_group`, where True means "artifact,
    keep regardless of length". `describe_image` returns prose for EVERY image, so a decorative
    selfie glossed "A man standing in front of a whiteboard" inferred substance=True and KEPT a
    post that a real cascade read (TYPE: PHOTO → text="" → substance=False) would have DROPPED.
    Same image, opposite decision, settled by which ingester touched the URL first.

    None means "cache miss" to every caller, so the entry is re-read (~$0.0001) and overwritten —
    the cache upgrades itself rather than serving a format it cannot interpret."""
    assert ocr_cascade.from_cache("an old describe_image gloss") is None
    assert ocr_cascade.from_cache("") is None
    assert ocr_cascade.from_cache(None) is None


# ── per-run model resolution (OCR_FALLBACKS walked once, substitutions LOGGED) ──────

def test_fallback_model_is_used_and_logged(monkeypatch):
    from pipeline import model_routing
    ocr_cascade._reset_stage_for_tests()
    fb = model_routing.OCR_FALLBACKS[1]
    monkeypatch.setattr(model_routing, "resolve_ocr_model",
                        lambda **kw: (fb, f"FALLBACK {fb} (1 orgs) — primary is unroutable"))
    logged, calls = [], []
    monkeypatch.setattr(ocr_cascade, "log", lambda m: logged.append(m))

    def fake(role, **kw):
        calls.append(kw["model"])
        return _Resp("row data " * 20 + "\nTYPE: SCREENSHOT")
    monkeypatch.setattr(ocr_cascade.llm_client, "call", fake)

    assert ocr_cascade.read_image("http://img") is not None
    assert calls == [fb]                                 # the RESOLVED model, not the constant
    assert any("FALLBACK" in m for m in logged)          # a silent swap is the banned failure
    ocr_cascade._reset_stage_for_tests()


def test_every_candidate_dead_disables_before_any_call(monkeypatch):
    from pipeline import model_routing
    ocr_cascade._reset_stage_for_tests()
    monkeypatch.setattr(model_routing, "resolve_ocr_model",
                        lambda **kw: (None, "no declared OCR model is routable under deny=['x']"))

    def bomb(role, **kw):
        raise AssertionError("paid a round-trip with no routable model")
    monkeypatch.setattr(ocr_cascade.llm_client, "call", bomb)

    assert ocr_cascade.read_image("http://img") is None
    assert "no declared OCR model is routable" in (ocr_cascade.stage_status() or "")
    ocr_cascade._reset_stage_for_tests()


def test_this_module_holds_no_price_table():
    """The OCR spend is recorded by `llm_client.call` from the charge OpenRouter REPORTS, keyed on
    the model override — so a table here would be a second, less accurate copy of it, and one that
    goes stale silently (see `llm_spend._PRICING`'s 5.5x-low llama row). Deleted 2026-08-28."""
    assert not hasattr(ocr_cascade, "cost") and not hasattr(ocr_cascade, "_PRICE")
