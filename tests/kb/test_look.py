"""`look`: search outside the library now for a few
named things, on the watch sources, and hand back what each found with the `hopper` arguments to
save it. The flow against fake sources, never the network; the tool through the real server."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server import build
from pipeline.frontier import lanes, look
from pipeline.frontier.candidates import Candidate
from tests import mcp_session

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


class _Source:
    """A source whose results name the spelling it was asked, newest last."""

    def __init__(self, slug, n=1, *, error=None, up=True, make=None):
        self.slug, self.host, self.min_interval_s = slug, slug, 0.0
        self._n, self._error, self._up, self._make = n, error, up, make
        self.asked = []

    def available(self):
        return self._up

    def search(self, query, *, since, limit, now, context=()):
        self.asked.append((query, since, list(context)))
        if self._error:
            raise self._error
        make = self._make or (lambda i: Candidate(
            candidate_id=f"{self.slug}:{query}:{i}", source=self.slug, kind="paper",
            title=f"{query} result {i}", url=f"https://example/{self.slug}/{i}",
            published=f"2026-10-{i + 1:02d}", summary=f"about {query}"))
        return [make(i) for i in range(self._n)]


def test_each_name_is_searched_on_every_source_now_and_comes_back_newest_first():
    registry = {"arxiv": _Source("arxiv", n=10), "x": _Source("x", n=2)}
    out = look.run(names=[{"name": "Chai-2", "spellings": ["Chai 2"], "context": ["antibody"]}],
                   now=NOW, registry=registry)
    assert out["status"] == "ok" and out["days"] == 7
    [chai] = out["names"]
    assert chai["by_source"] == {"arxiv": 20, "x": 4}            # both spellings, distinct finds
    assert [r["when"] for r in chai["results"]["arxiv"]] == [
        "2026-10-10", "2026-10-10", "2026-10-09", "2026-10-09", "2026-10-08", "2026-10-08",
        "2026-10-07", "2026-10-07"]                               # newest first, SHOWN of them
    assert len(chai["results"]["arxiv"]) == look.SHOWN
    assert registry["x"].asked == [("Chai-2", NOW - timedelta(days=7), ["antibody"]),
                                   ("Chai 2", NOW - timedelta(days=7), ["antibody"])]
    first = chai["results"]["x"][0]
    assert first["cite"].startswith("[Chai-2 result 1 · 2026-10-02](https://example/x/1)")
    assert first["save"] == {"reference": "https://example/x/1", "confirm": True}
    assert "never present a result as theirs" in out["next"]


def test_a_source_that_cannot_be_asked_or_fails_says_so_and_the_rest_still_answer():
    registry = {"x": _Source("x", up=False), "openalex": _Source("openalex", error=RuntimeError()),
                "github": _Source("github", n=0)}
    out = look.run(names=["x402"], days=3, now=NOW, registry=registry)
    assert out["names"][0]["by_source"] == {"x": "unavailable", "openalex": "failed",
                                            "github": 0}
    assert out["names"][0]["results"] == {} and registry["x"].asked == []
    assert registry["github"].asked[0][1] == NOW - timedelta(days=3)


def test_a_find_that_does_not_name_it_is_not_shown():
    """The daily pulls' name check, the same one: look and a watch never disagree."""
    stray = _Source("arxiv", make=lambda i: Candidate(
        candidate_id="arxiv:1", source="arxiv", kind="paper", title="Something else",
        url="https://example/1"))
    assert look.run(names=["Boltz-2"], now=NOW,
                    registry={"arxiv": stray})["names"][0]["by_source"] == {"arxiv": 0}


def test_substack_results_say_how_to_save_them_and_a_paid_one_that_it_cannot_be():
    def post(i):
        url = ("https://bio.substack.com/p/a", "https://www.latent.space/p/b",
               "https://www.forever.ai/p/c")[i]
        return Candidate(candidate_id=f"substack:{i}", source="substack", kind="substack",
                         title="Pub", url=url, published=f"2026-10-1{i}",
                         summary="names rentosertib", payload={"paid": i == 2})
    out = look.run(names=["rentosertib"], now=NOW,
                   registry={"substack": _Source("substack", n=3, make=post)})
    paid, custom, hosted = out["names"][0]["results"]["substack"]       # newest first
    assert paid["paid"] is True and "save" not in paid
    assert custom["save"] == {"reference": "https://www.latent.space/p/b", "confirm": True,
                              "kind_hint": "substack"}
    assert hosted["save"] == {"reference": "https://bio.substack.com/p/a", "confirm": True}


def test_a_window_of_years_is_served_whole_by_the_sources_that_can_and_narrowed_by_those_that_cannot():
    """A literature window: OpenAlex takes all five years, Substack is asked its own month and
    the answer says so, so the host cannot report a silence it never looked for."""
    registry = {"openalex": _Source("openalex"), "substack": _Source("substack")}
    registry["substack"].max_window_days = 30
    out = look.run(names=["morpholine"], days=1825, now=NOW, registry=registry)
    assert out["status"] == "ok" and out["days"] == 1825
    assert registry["openalex"].asked[0][1] == NOW - timedelta(days=1825)
    assert registry["substack"].asked[0][1] == NOW - timedelta(days=30)
    assert out["searched_less"] == {"substack": 30}
    assert "substack over 30 days" in out["next"] and "you did not look" in out["next"]


def test_a_long_windows_results_keep_the_order_the_source_ranked_them_in():
    """Asked for a landscape, OpenAlex answers most-cited first — re-sorting that page by date
    would cut the field's most-cited work out of the `SHOWN` the host ever sees."""
    registry = {"openalex": _Source("openalex", n=10)}
    out = look.run(names=["morpholine"], days=1825, now=NOW, registry=registry)
    assert [r["when"] for r in out["names"][0]["results"]["openalex"]] == [
        "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06",
        "2026-10-07", "2026-10-08"]                       # as returned, oldest first here
    short = look.run(names=["morpholine"], days=90, now=NOW, registry=registry)
    assert short["names"][0]["results"]["openalex"][0]["when"] == "2026-10-10"   # newest first


def test_a_window_every_source_can_serve_says_nothing_about_windows():
    out = look.run(names=["Boltz-2"], days=7, now=NOW, registry={"arxiv": _Source("arxiv")})
    assert "searched_less" not in out and "cannot be searched that far back" not in out["next"]


@pytest.mark.parametrize("args, says", [
    ({"names": []}, "Name 1 to 3"),
    ({"names": ["a", "b", "c", "d"]}, "Name 1 to 3"),
    ({"names": ["x402"], "sources": ["reddit"]}, "Unknown sources reddit"),
    ({"names": ["x402"], "days": 0}, "1 or more"),
    ({"names": ["x402"], "days": "a week"}, "1 or more"),
    ({"names": [{"name": "--"}]}, "names[0] needs a name"),
])
def test_a_refused_look_searches_nothing(args, says):
    source = _Source("arxiv")
    out = look.run(now=NOW, registry={"arxiv": source}, **{"sources": None, **args})
    assert out["status"] == "refused" and says in out["reason"]
    assert source.asked == []


def test_a_card_shows_who_or_what_when_its_link_and_the_opening_of_its_text():
    c = Candidate(candidate_id="x:1", source="x", kind="x", title="Lab (@lab)",
                  url="https://x.com/lab/status/1", published="2026-10-09T13:55:36+00:00",
                  summary="y" * 500)
    assert lanes.card(c) == {"title": "Lab (@lab)", "when": "2026-10-09",
                             "cite": "[Lab (@lab) · 2026-10-09](https://x.com/lab/status/1)",
                             "snippet": "y" * lanes.SNIPPET_CHARS}


# ── the tool ───────────────────────────────────────────────────────────────────────
def test_look_is_read_only_and_reaches_the_flow(kb_home, monkeypatch):
    [tool] = [t for t in asyncio.run(build.server().list_tools()) if t.name == "look"]
    declared = tool.annotations.model_dump(by_alias=True)
    assert (declared["readOnlyHint"], declared["destructiveHint"],
            declared["openWorldHint"]) == (True, False, True)
    monkeypatch.setattr(look.adapters, "adapters", lambda: {"arxiv": _Source("arxiv")})
    out = mcp_session.session().call("look", names=["Boltz-2"])
    assert out["status"] == "ok" and out["names"][0]["by_source"] == {"arxiv": 1}
