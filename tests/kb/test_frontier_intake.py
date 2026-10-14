"""`pipeline/frontier/intake.py` — a find is kept only when it names what the watch watches."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.frontier import intake


def _c(cid, title, summary=""):
    return SimpleNamespace(candidate_id=cid, title=title, summary=summary)


@pytest.mark.parametrize("name, text, held", [
    ("Boltz-2", "Benchmarking Boltz-2 on kinase affinity", True),
    ("Boltz2", "UniversalBasicInc/boltz2-intel-arc", True),          # a repo's name
    ("Boltz-2", "UniversalBasicInc/boltz2-intel-arc", False),        # another spelling: its own watch
    ("Virtual Cell Challenge", "Results of the virtual cell challenges", True),   # a plural
    ("Virtual Cell Challenge", "A virtual staining challenge for cell images", False),
    ("rentosertib", "Incentivizing challenging drug discovery", False),
    ("x402", "Settling agent payments over x402", True),
    ("GDPval", "", False),
    ("", "anything at all", False),
])
def test_a_name_counts_only_as_a_phrase(name, text, held):
    assert intake.names(name, text) is held


def test_a_find_is_kept_when_its_title_or_summary_names_the_watch():
    page = [_c("a", "Boltz-2 binding affinity at scale"),
            _c("b", "A docking survey", "We compare against Boltz-2 and Chai-1."),
            _c("c", "The eye as a canary in the coal mine")]
    kept, dropped = intake.named(page, "Boltz-2")
    assert [c.candidate_id for c in kept] == ["a", "b"] and dropped == 1
