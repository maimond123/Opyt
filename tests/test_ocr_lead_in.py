"""The chart read's "Here's a ...:" lead-in is removed wherever a MediaRead is built.

Scope note, so nobody later reads these tests as a retrieval fix: this is HYGIENE. Measured over
all 105 affected chunks on 2026-08-10, removing the lead-in moved their cross-atom cosine p50 from
0.433 to 0.420. It does not break false region binding, and nothing here claims it does.
"""
from __future__ import annotations

import pytest

from pipeline import ocr_cascade as oc

LEAD_INS = [
    "Here's a detailed extraction of information from the provided chart:",
    "Here's a concrete extraction of information from the provided chart:",
    "Here's a detailed breakdown of the provided diagram:",
    "Here's an analysis of the provided bar chart:",
    "Here is a summary of the chart:",
    "Here’s a detailed breakdown of the chart:",          # curly apostrophe, as the model emits it
]


@pytest.mark.parametrize("lead", LEAD_INS)
def test_every_observed_lead_in_form_is_removed(lead):
    body = "**Title:** Wage Growth\n**Y-axis:** Avg. weekly wages"
    assert oc._strip_lead_in(f"{lead}\n\n{body}") == body


def test_the_chart_content_itself_survives():
    """The title is the chart's, not the sentence's — losing it would defeat the whole VLM hop."""
    out = oc._strip_lead_in(
        "Here's a detailed extraction of information from the provided chart:\n\n"
        "**Title:** AI Agent Development Has Reached an Inflection Point\n"
        "**X-axis:** Time (Years)")
    assert "AI Agent Development Has Reached an Inflection Point" in out
    assert "X-axis" in out
    assert "Here's" not in out


def test_prose_that_merely_contains_a_colon_is_untouched():
    """The regex is anchored and length-bounded so it cannot eat real content.

    This is the failure that would matter: a chart read whose FIRST line is substance whose text
    happens to contain a colon. Silently deleting it would be indistinguishable from a bad VLM read.
    """
    for keep in (
        "The chart shows three regimes: pre-2020, 2020-2023, and after.",
        "Revenue: $2.31M in Q3, up from $1.80M.",
        "Here the trend reverses after 2021 with no colon at all",
        "A long preamble that runs well past the hundred and twenty character bound this regex "
        "allows before it ever reaches its terminating colon, so it must not match:",
    ):
        assert oc._strip_lead_in(keep) == keep


def test_strip_is_idempotent_and_empty_safe():
    once = oc._strip_lead_in("Here's a breakdown of the chart:\n\n**Title:** X")
    assert oc._strip_lead_in(once) == once
    assert oc._strip_lead_in("") == ""


def test_a_media_read_normalizes_on_construction():
    """Direct construction — the `read_image` path."""
    m = oc.MediaRead("Here's an analysis of the provided chart:\n\n**Title:** Q3", "chart", True)
    assert m.text == "**Title:** Q3"


def test_a_cached_entry_written_before_this_existed_is_normalized_on_rehydration():
    """The discriminating case for putting the strip in `__post_init__`.

    `image_descriptions.json` is keyed by immutable CDN URL and holds entries written months ago,
    every one of them carrying the lead-in. If the strip lived only in `read_image`, a cache HIT
    would serve the old text forever and the corpus would stay half-clean — which is exactly the
    two-shapes-in-one-keyspace bug `from_cache` already exists to prevent.
    """
    stale = {"text": "Here's a detailed extraction of information from the provided chart:\n\n"
                     "**Title:** Above-Trend Wage Growth",
             "kind": "chart", "substance": True}
    m = oc.from_cache(stale)
    assert m is not None
    assert m.text == "**Title:** Above-Trend Wage Growth"


def test_to_cache_round_trips_the_normalized_form():
    """A normalized read written back to the cache stays normalized — the cache upgrades in place."""
    m = oc.MediaRead("Here's a breakdown of the chart:\n\n**Title:** X", "chart", True)
    assert oc.from_cache(m.to_cache()).text == "**Title:** X"


def test_the_chart_prompt_forbids_the_lead_in():
    """The strip cleans what exists; the prompt is what stops new ones being produced."""
    assert "Do not open with a sentence about what you are about to do." in oc._CHART_PROMPT
