"""The four library writes as the web gives them: what is answered for each way a call can stand,
and the local tools' own answers with the lines for the Mac's host left out.

`answered` is a decision, checked with plain values. The local answers are made here by the local
tools themselves, registered as the Mac's server registers them over a seeded store of their own,
so a line a local tool starts writing for its own host fails here until the web's answer drops it.
"""
from __future__ import annotations

import json
import re

import pytest

from mcp_server.forget_tools import register_forget_tools
from mcp_server.hopper_tools import register_hopper_tools
from mcp_server.oracle_tools import register_oracle_tools
from mcp_server.topic_tools import register_topic_tools
from pipeline.kb import schema
from pipeline.oracles import reference
from pipeline.platforms import connections
from pipeline.worker import liveness
from service import library_writes, line
from tests.kb.test_export import _corpus

# How a write is done, and the local tools the web does not have. `latest` is a web tool, but its
# topics are not, and no answer here is about anything else of it.
HOW = re.compile(r"\bMacs?\b|queue|\bthe line\b|check[- ]in|onboard|(?<![\w`])oracle\("
                 r"|\blook\(|worker|background|latest|\binstall|served copy|\bpush\b"
                 r"|delete_atoms|source=|checked")


def texts(value) -> list[str]:
    """Every string in an answer."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in texts(v)]
    if isinstance(value, list):
        return [t for v in value for t in texts(v)]
    return []


def done(tool: str, args: dict, got: dict) -> dict:
    """The web's answer for a call the Mac answered with `got`."""
    return library_writes.answered(tool, args, line.Result(line.DONE, got))


class Registry:
    """What the local tools register with: each tool's function by its name."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kw):
        def keep(fn):
            self.tools[fn.__name__] = fn
            return fn
        return keep


@pytest.fixture()
def local(tmp_path, monkeypatch):
    """`local(tool, **args)`: the answer the Mac would post, from the local tool itself, over the
    wire's JSON. The install shares, and its worker says the pulls are running, which is when
    the local answers say the most about the Mac."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "mac"))
    monkeypatch.setenv("OPYT_SERVICE_TOKEN", "an-owner-token")
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda *a, **k: True)
    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {"ok": True})
    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    conn = schema.connect()
    _corpus(conn)
    conn.close()
    registry = Registry()
    for register in (register_hopper_tools, register_forget_tools, register_topic_tools,
                     register_oracle_tools):
        register(registry)
    return lambda tool, **args: json.loads(json.dumps(registry.tools[tool](**args)))


# ── what is answered, by how the call stands ───────────────────────────────────────────

@pytest.mark.parametrize("tool, args", [
    ("hopper", {"reference": "https://example.com/a", "confirm": True}),
    ("forget", {"atom_id": "x:1", "confirm": True}),
    ("watch", {"action": "create", "name": "Boltz-2"}),
    ("watch", {"action": "edit", "topic_id": "t_1"}),
    ("watch", {"action": "drop", "topic_id": "t_1"}),
    ("add_oracle", {"reference": "@karpathy", "confirm": True}),
])
def test_a_change_is_being_made_or_will_be_and_only_that_sentence_names_the_computer(tool, args):
    working = library_writes.answered(tool, args, line.Result(line.RUNNING, None))
    later = library_writes.answered(tool, args, line.Result(line.WAITING, None))
    assert working["status"] == "working" and "computer" not in working["message"]
    assert later["status"] == "later"
    assert later["message"].endswith("when your computer is next on.")
    for answer in (working, later):
        assert set(answer) == {"status", "message"} and not HOW.search(answer["message"])


@pytest.mark.parametrize("action", ["list", "preview"])
def test_a_list_or_a_preview_has_nothing_to_do_later(action):
    args = {"action": action}
    assert library_writes.answered("watch", args, line.Result(line.WAITING, None)) == {
        "status": "unavailable", "message": "Topics can't be shown right now."}
    working = library_writes.answered("watch", args, line.Result(line.RUNNING, None))
    assert working["status"] == "working" and "`status`" in working["message"]


def test_a_failed_call_gives_the_macs_reason_or_the_tools_own_sentence():
    args = {"reference": "https://example.com/a", "confirm": True}
    assert library_writes.answered("hopper", args, line.Result(
        line.FAILED, {"message": "That page could not be reached."})) == {
            "status": "failed", "message": "That page could not be reached."}
    assert library_writes.answered("hopper", args, line.Result(line.FAILED, {})) == {
        "status": "failed", "message": "The link could not be saved."}


# ── the local tools' own answers, as the web gives them ────────────────────────────────

def test_a_saved_link_is_answered_as_the_local_tool_answers_it(local):
    for link in ("https://github.com/root/agentkit", "not a link",
                 "https://www.youtube.com/watch?v=abc"):
        args = {"reference": link, "confirm": True}
        got = local("hopper", **args)
        assert got["status"] in ("already_present", "unroutable")
        assert done("hopper", args, got) == got
        assert not any(HOW.search(t) for t in texts(got)), got


def test_a_removed_item_keeps_the_local_consent_without_the_served_copy_sentence(local):
    args = {"atom_id": "x:1", "confirm": True}
    preview = local("forget", atom_id="x:1")
    got = local("forget", **args)
    assert got["status"] == "forgotten" and got["shared"] is True
    assert "served copy" in got["consent"][-1], "what the Mac's own host is told"

    said = done("forget", args, got)
    assert said == {"status": "forgotten", "scope": "atom", "atom_id": "x:1",
                    "description": "x:1 card", "who_id": "x:user:1", "when_ts": "2025-01-15",
                    "source_url": "https://example/x:1", "consent": preview["consent"][:-1]}
    assert any("bookmarks" in s for s in said["consent"]), "how it can come back is still said"
    assert not any(HOW.search(t) for t in texts(said)), said


def test_several_removed_items_are_answered_the_same_way(local):
    args = {"atom_ids": ["x:2", "blog:home"], "confirm": True}
    said = done("forget", args, local("forget", **args))
    assert said["status"] == "forgotten" and [a["atom_id"] for a in said["atoms"]] == [
        "x:2", "blog:home"]
    assert said["consent"][0] == "This removes these 2 atoms."
    assert set(said) == {"status", "scope", "atoms", "consent"}
    assert not any(HOW.search(t) for t in texts(said)), said

    missing = {"atom_ids": ["x:1", "nope:1"], "confirm": True}
    refused = local("forget", **missing)
    assert refused["status"] == "not_found" and done("forget", missing, refused) == refused


def test_an_oracle_no_longer_followed_is_answered_with_the_sentences_true_here(local):
    args = {"oracle": "@alice", "confirm": True}
    got = local("forget", **args)
    assert got["status"] == "forgotten" and "delete_atoms" in " ".join(got["consent"])

    said = done("forget", args, got)
    assert said == {
        "status": "forgotten", "scope": "oracle", "canonical_id": "x:user:1", "name": "Alice",
        "atoms_kept": 1,
        "consent": ["This stops tracking one Oracle: Alice (x:user:1).",
                    "Their 1 existing atoms, identity records and trust roots remain.",
                    "Future scheduled refreshes will not select this person. Removal takes "
                    "effect at once: whatever a running refresh brings in after it is not kept. "
                    "Bookmarks and separately saved links can still arrive.",
                    "Adding this Oracle again starts a new subscription."]}
    assert set(said["consent"]) < {s if "delete_atoms" not in s else s.split(" Remove")[0]
                                   for s in got["consent"]}, "each is the local tool's own"
    assert not any(HOW.search(t) for t in texts(said)), said

    nobody = {"oracle": "nobody we follow", "confirm": True}
    assert done("forget", nobody, local("forget", **nobody))["status"] == "not_found"


def test_a_new_topic_is_answered_without_the_lines_about_finds_in_latest(local):
    args = {"action": "create", "name": "Protein design",
            "watches": [{"name": "Boltz-2", "sources": ["github"]}], "words": ["protein design"]}
    got = local("watch", **args)
    assert got["status"] == "ok" and "latest" in got["first"] and "latest" in got["next"]

    said = done("watch", args, got)
    assert said == {**{k: v for k, v in got.items() if k not in ("first", "next")},
                    "message": library_writes._FOLLOWED}
    assert not any(HOW.search(t) for t in texts(said)), said

    for action, more in (("list", {}), ("edit", {"topic_id": got["topic_id"], "name": "Proteins"}),
                         ("drop", {"topic_id": got["topic_id"]}),
                         ("drop", {"topic_id": "t_none"}), ("create", {"name": ""})):
        changed = local("watch", action=action, **more)
        assert done("watch", {"action": action, **more}, changed) == changed
        assert not any(HOW.search(t) for t in texts(changed)), changed


def test_a_preview_is_answered_without_the_local_hosts_check_of_the_finds(local):
    args = {"action": "preview", "name": "Protein design", "words": ["protein design"]}
    got = local("watch", **args)
    assert got["status"] == "preview" and "check its finds" in got["next"]

    said = done("watch", args, got)
    assert said == {**{k: v for k, v in got.items() if k != "next"},
                    "message": library_writes._PREVIEWED}
    assert not any(HOW.search(t) for t in texts(said)), said


def test_an_added_oracle_is_answered_without_where_the_pull_stands_on_the_mac(local):
    args = {"reference": "@alice", "confirm": True}
    got = local("add_oracle", **args)
    assert "Keep your Mac open" in got["message"] and "oracle(action='progress')" in got["message"]

    said = done("add_oracle", args, got)
    assert said == {"status": "added", "name": "Alice", "canonical_id": "x:user:1",
                    "was_already_oracle": True}


def test_a_person_who_cannot_be_added_is_answered_with_the_local_reason(local):
    args = {"reference": "https://github.com/karpathy", "confirm": True}
    got = local("add_oracle", **args)
    assert done("add_oracle", args, got) == {"status": "not_added", "message": got["error"]}
    bad_window = {"reference": "@alice", "confirm": True, "x_lookback": "1y"}
    assert done("add_oracle", bad_window, local("add_oracle", **bad_window))["status"] == "not_added"
    for url in ("https://github.com/karpathy", "https://x.com/karpathy",
                "https://scholar.google.com/citations?user=abc", "https://www.youtube.com/@x"):
        assert not HOW.search(reference.unsupported_root(url) or ""), url


# ── an outcome nobody waited for, in one line ──────────────────────────────────────────

@pytest.mark.parametrize("tool, args, state, got, line_said", [
    ("hopper", {"reference": "https://a.example"}, line.DONE, {"status": "saved"},
     "https://a.example is saved in the library."),
    ("hopper", {"reference": "https://a.example"}, line.DONE, {"status": "already_present"},
     "https://a.example was already in the library."),
    ("hopper", {"reference": "https://a.example"}, line.DONE,
     {"status": "failed", "detail": "could not fetch or identify that as a paper."},
     "https://a.example was not saved: could not fetch or identify that as a paper."),
    ("hopper", {"reference": "https://a.example"}, line.FAILED, {},
     "https://a.example was not saved: The link could not be saved."),
    ("forget", {"atom_id": "x:1"}, line.DONE,
     {"status": "forgotten", "scope": "atom", "atom_id": "x:1", "description": "A post on rollups"},
     "A post on rollups is removed from the library."),
    ("forget", {"atom_ids": ["x:1", "x:2"]}, line.DONE,
     {"status": "forgotten", "scope": "atoms", "atoms": [{"atom_id": "x:1"}, {"atom_id": "x:2"}]},
     "2 items are removed from the library."),
    ("forget", {"atom_id": "x:1"}, line.DONE, {"status": "not_found", "scope": "atom"},
     "x:1 was not found in the library, so nothing was removed."),
    ("forget", {"oracle": "@alice"}, line.DONE,
     {"status": "forgotten", "scope": "oracle", "canonical_id": "x:user:1", "name": "Alice",
      "atoms_kept": 1, "consent": []},
     "Alice is no longer followed."),
    ("forget", {"oracle": "Sam"}, line.DONE,
     {"status": "ambiguous", "message": "Choose one canonical_id and preview it again."},
     "Sam was not removed: Choose one canonical_id and preview it again."),
    ("watch", {"action": "create", "name": "Boltz-2"}, line.DONE,
     {"status": "ok", "name": "Boltz-2", "first": "…", "next": "…"}, "The topic Boltz-2 is followed."),
    ("watch", {"action": "edit", "topic_id": "t_1"}, line.DONE, {"status": "ok", "name": "Boltz-2"},
     "The topic Boltz-2 is changed."),
    ("watch", {"action": "drop", "topic_id": "t_1"}, line.DONE,
     {"status": "ok", "dropped": "Boltz-2"}, "The topic Boltz-2 is no longer followed."),
    ("watch", {"action": "create"}, line.DONE,
     {"status": "refused", "reason": "A topic needs a name: the subject in the user's words."},
     "The topic was not changed: A topic needs a name: the subject in the user's words."),
    ("watch", {"action": "preview", "name": "Boltz-2"}, line.DONE, {"status": "preview", "next": "…"},
     "The preview of Boltz-2 is ready: `status` returns it."),
    ("watch", {"action": "list"}, line.FAILED, {"message": "It could not be read."},
     "The list of topics did not come: It could not be read."),
    ("add_oracle", {"reference": "@karpathy"}, line.DONE,
     {"added": {"canonical_id": "x:user:9", "name": "Andrej Karpathy",
                "was_already_oracle": False}},
     "Andrej Karpathy is added as an Oracle."),
    ("add_oracle", {"reference": "@nobody"}, line.DONE,
     {"error": "could not resolve '@nobody' — check the handle/URL. Nothing was written."},
     "@nobody was not added: could not resolve '@nobody' — check the handle/URL. Nothing was "
     "written."),
])
def test_an_outcome_nobody_waited_for_is_one_plain_line(monkeypatch, tool, args, state, got,
                                                        line_said):
    ended = [{"tool": tool, "args": args, "state": state, "answer": got}]
    monkeypatch.setattr(line, "untold", lambda owner: ended)
    assert library_writes.earlier("a-library") == [line_said]
    assert not HOW.search(line_said)
