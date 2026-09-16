"""The 60-second wall — what a host may say when a call comes back as an error.

MEASURED 2026-09-14, one full onboarding run through Claude Desktop: nine
`oracle(action='ingest')` calls, SIX killed at the client's wall. Each died holding the only thing
the user was meant to hear — `presentation`, `partial_note`, the whole §G vocabulary — so the model
received exactly one fact, `Error: Request timed out`, and used it as a universal explanation for
three things that were not timeouts at all. Four retries fired off the back of that; none helped.
Design record: docs/plans/2026-09-14-sixty-second-wall.md §B, §C.

⚠️ THESE RULES CANNOT LIVE IN A TOOL RETURN, which is the whole asymmetry of §B: the return is
precisely what a truncated call destroys. A tool DESCRIPTION is delivered at tool-list time and is
still in context at the one moment it is needed — when the call comes back as an error.

The §A half of that plan (a foreground time budget) was built, measured on a second live run, and
REVERTED — it did not reliably beat the wall, because R4 keeps a blog archive running to
completion once started and one archive measured most of sixty seconds on its own. See
`2026-09-14-sixty-second-wall-landed.md`. What survived is this, and the payload bound in
`tests/kb/test_screen.py`.
"""
from __future__ import annotations

from mcp_server import oracle_tools


def _oracle_tool_doc() -> str:
    captured: dict = {}

    class _FakeMCP:
        def tool(self):
            def _wrap(fn):
                captured.setdefault(fn.__name__, fn.__doc__ or "")
                return fn
            return _wrap

    oracle_tools.register_oracle_tools(_FakeMCP())
    return captured["oracle"]


def test_the_no_retry_rule_lives_in_the_tool_description():
    """Where it has to live. A return value is unreachable exactly when this matters."""
    doc = _oracle_tool_doc()

    assert "Do not start another ingest" in doc
    assert 'Never say "timed out"' in doc
    assert "Never explain an absence you did not measure" in doc


def test_the_rule_forbids_a_SMALLER_retry_and_not_just_the_same_one():
    """⚠️ THE WORDING IS THE FIX. The first version said "do not call it again", and on
    2026-09-14 20:05:53 a model read that as being about the SAME call, reasoned "I\'ll run the
    pull one person at a time to stay under the time limit", and retried. Calls to this server
    serialize, so the smaller second call waits out the first and then times out having done
    less — measured at 125s, twice. Fewer calls, never smaller ones."""
    doc = _oracle_tool_doc()

    assert "NOT a smaller one" in doc
    assert "one at a time" in doc


def test_the_description_never_offers_the_client_timeout_knob():
    """Raising `mcpToolTimeoutSec` is machine-wide admin policy, does not exist on every client,
    and a build that needs it violates the distributable invariant. It is not a mitigation and
    must never be suggested as one. `.guards.py` enforces this repo-wide."""
    assert "mcpToolTimeoutSec" not in _oracle_tool_doc()
