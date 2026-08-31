"""Spend must come from what the provider CHARGED, not from a per-model rate card.

Why this file exists. The static `_PRICING` table was correct while OpenRouter's default routing
picked the cheapest upstream — the table simply held the cheapest upstream's price. Ranking by
`provider.sort: "throughput"` broke that assumption at the root: the serving upstream is chosen per
REQUEST and the upstreams do not share a rate card (llama-3.3-70b spans $0.10/M on DeepInfra to
$1.04/M on Together), so price stopped being a property of the model id.

Measured, not assumed — 12 real content_quality calls, 2026-07-31, against the table as it stood
that day (llama-3.3-70b at $0.13/$0.40 per M):
    actually billed (usage.cost)   $0.00119/call   ->  $0.0143
    what cost_for() reported       $0.00035/call   ->  $0.0042      3.4x understated

The understatement was not cosmetic: `spend_total()` is the lifetime figure every cost ceiling
reads, so an under-reported total keeps spending real money past a ceiling the user set.

That 3.4x is a DATED measurement, not a standing property, and the tests below must not assert it.
The llama row was corrected to the catalog's $0.71/$0.71 on 2026-08-28, since when the table reads
HIGH on this sample. The gap's direction moves with whichever upstream served the request and with
how stale the row is; only "the reported charge wins" is invariant.

The table stays as the FALLBACK — Anthropic reports no per-call cost, and a malformed or absent
usage block must not crash or zero out accounting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline import llm_client, llm_spend


# ── helpers ──────────────────────────────────────────────────────────────────────


class _PassthroughBreaker:
    """The real breaker persists state in opyt.db; tests must not touch it.

    Signature MIRRORS CircuitBreaker.call, `ignore` included: a double that drifts from the real
    interface stops being a stand-in and starts being a second, wrong implementation."""

    def call(self, fn, *, ignore: tuple = ()):
        try:
            return fn()
        except ignore:
            raise


@pytest.fixture
def isolated_stats(tmp_path: Path):
    stats_path = tmp_path / "api_stats.json"
    llm_client._override_stats_file_for_tests(stats_path)
    yield stats_path
    llm_client._override_stats_file_for_tests(None)


@pytest.fixture
def openrouter_reporting(monkeypatch):
    """Run the REAL _call_openrouter_sync against a canned response, so the assertions cover the
    body we put on the wire AND the usage block we read back off it."""
    seen: dict = {}

    def fake_http_json(req, timeout=180.0):
        seen["body"] = json.loads(req.data.decode())
        return ({"choices": [{"message": {"content": "ok"}}],
                 "provider": "SambaNova",
                 "usage": {"prompt_tokens": 2503, "completion_tokens": 69,
                           "cost": 0.00118845}}, 0.01)

    monkeypatch.setattr(llm_client, "_http_json", fake_http_json)
    monkeypatch.setattr(llm_client, "_openrouter_breaker", lambda: _PassthroughBreaker())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return seen


def _backend_returning(raw: dict | None, in_tok: int = 2503, out_tok: int = 69):
    """A backend fake in the 5-tuple contract, with a caller-chosen raw response."""
    def _fn(model, system, user, max_tokens, response_format=None, images=None):
        return "ok", in_tok, out_tok, 0.42, raw
    return _fn


@pytest.fixture
def backend():
    """Install a backend fake mid-test, restoring the real one at teardown. `real` is captured
    BEFORE any install, so a test that installs twice still restores the genuine backend — leaking
    a fake here would silently disarm every later test in the session."""
    real = llm_client._BACKENDS["openrouter"]

    def _install(raw):
        llm_client._set_backend_for_tests("openrouter", _backend_returning(raw))

    yield _install
    llm_client._set_backend_for_tests("openrouter", real)


# ── reading the reported charge ──────────────────────────────────────────────────


def test_reads_usage_cost():
    """USD, verified against OpenRouter's own rate card rather than assumed: 2503 prompt tokens on
    SambaNova ($0.45/M) reported 0.00112635 as the prompt component."""
    assert llm_client._reported_cost({"usage": {"cost": 0.00118845}}) == pytest.approx(0.00118845)


@pytest.mark.parametrize("raw", [
    None,                                  # backend supplied no response at all
    {},                                    # no usage block (Anthropic's shape)
    {"usage": None},
    {"usage": {"prompt_tokens": 10}},      # usage present, cost absent
    {"usage": {"cost": "not-a-number"}},   # malformed
    "not-a-dict",
])
def test_unreported_falls_back(raw):
    """None means 'use the table'. Accounting must degrade to an estimate, never crash and never
    silently record zero."""
    assert llm_client._reported_cost(raw) is None


def test_zero_is_treated_as_unreported():
    """Under BYOK, OpenRouter charges nothing and the real money is spent upstream, so a literal
    0.0 would under-report in exactly the way the static table did. A genuinely free model is
    absent from _PRICING, so falling back still yields 0.0 — one rule covers both."""
    assert llm_client._reported_cost({"usage": {"cost": 0.0}}) is None


# ── what `call()` records ────────────────────────────────────────────────────────


def test_call_prefers_the_reported_charge(backend, isolated_stats):
    """THE regression guard: the recorded figure is what the provider says it CHARGED, whatever
    the table would have estimated."""
    backend({"usage": {"cost": 0.00118845}})
    resp = llm_client.call("content_quality", system="s", user="u")
    assert resp.cost_usd == pytest.approx(0.00118845)

    # The DIRECTION the table errs is a dated measurement, not the contract, so it is not asserted:
    # it read 3.4x LOW while the llama row was stale and reads high since the 2026-08-28 price
    # correction. What must hold is that the table figure is not the one recorded.
    table = llm_client.cost_for("meta-llama/llama-3.3-70b-instruct", 2503, 69)
    assert resp.cost_usd != pytest.approx(table), "recorded the rate-card estimate"


def test_call_falls_back_to_the_table(backend, isolated_stats):
    """Anthropic returns no per-call cost, so the rate card is still the only number available."""
    backend({"usage": {"prompt_tokens": 2503, "completion_tokens": 69}})
    resp = llm_client.call("content_quality", system="s", user="u")
    assert resp.cost_usd == pytest.approx(
        llm_client.cost_for("meta-llama/llama-3.3-70b-instruct", 2503, 69))
    assert resp.cost_usd > 0, "fallback must still produce a number, not zero out accounting"


def test_spend_total_reflects_the_real_charge(backend, isolated_stats):
    """`spend_total()` is the lifetime spend figure every cost ceiling reads — an under-reported
    total keeps spending real money past a ceiling the user set."""
    backend({"usage": {"cost": 0.001}})
    before = llm_client.spend_total()
    for _ in range(5):
        llm_client.call("content_quality", system="s", user="u")
    assert llm_client.spend_total() - before == pytest.approx(0.005, rel=1e-6)


def test_record_external_cost_folds_into_lifetime(isolated_stats):
    """NON-LLM paid calls must land in the SAME lifetime total, or a ceiling reading `spend_total()`
    is blind to them. twitterapi is the live caller (pipeline/ingestion/x_render.py), hosted
    embedding the other (pipeline/kb/embed.py). This test used to live with the discover cadence
    and moved here when that was retired (2026-08-05): the cadence was one consumer of this
    number, never its owner, so its test belongs with the total."""
    assert llm_client.spend_total() == 0.0
    llm_client.record_external_cost("openrouter-embed", 0.003)
    llm_client.record_external_cost("openrouter-embed", 0.003, requests=1)
    assert llm_client.spend_total() == pytest.approx(0.006, abs=1e-9)
    assert llm_client._STATS["by_provider"]["openrouter-embed"] == {"requests": 2, "cost_usd": 0.006}
    llm_client.record_external_cost("openrouter-embed", 0.0)          # non-positive → no-op
    assert llm_client.spend_total() == pytest.approx(0.006, abs=1e-9)


# ── per-rail attribution ─────────────────────────────────────────────────────────
#
# Until 2026-08-16 ORACLE_REFRESH_DAILY_USD and BOOKMARK_CATCHUP_DAILY_USD were both $1.00 and both
# gated on `spend_today()` — the TOTAL — so the two rails shared ONE pool and whichever ran first
# spent the other's allowance. Worse, `frontier_execute` and `hopper` spent into that same total
# while checking nothing, so an UNGATED rail could exhaust the pool and a GATED one would be what
# stopped. These tests pin the attribution that lets each ceiling govern its own rail.


def test_spend_is_attributed_to_the_active_rail(isolated_stats):
    with llm_client.rail("bookmark_catchup"):
        llm_client.record_external_cost("openrouter-embed", 0.25)
    assert llm_client.spend_today_for_rail("bookmark_catchup") == 0.25
    assert llm_client.spend_today_for_rail("oracle_refresh") == 0.0
    assert llm_client.spend_today() == 0.25          # the total is unchanged


def test_spend_outside_any_rail_is_unattributed_not_dropped(isolated_stats):
    llm_client.record_external_cost("openrouter-embed", 0.10)
    assert llm_client.spend_today_for_rail("unattributed") == 0.10
    assert llm_client.spend_today() == 0.10          # ⚠️ never silently lost


def test_the_label_survives_a_threadpool(isolated_stats):
    """⚠️ THE TRAP. embed.py and x_render.py both fan out through a ThreadPoolExecutor, and
    a ContextVar does NOT propagate into one. Those are the two hottest spend paths, so a
    mechanism that fails here fails exactly where it matters and still looks correct in the
    totals."""
    from concurrent.futures import ThreadPoolExecutor
    with llm_client.rail("frontier_execute"):
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda _: llm_client.record_external_cost("openrouter-embed", 0.01), range(4)))
    assert llm_client.spend_today_for_rail("frontier_execute") == 0.04
    assert llm_client.spend_today_for_rail("unattributed") == 0.0


def test_the_scope_restores_the_previous_label(isolated_stats):
    with llm_client.rail("a"):
        with llm_client.rail("b"):
            assert llm_client.current_rail() == "b"
        assert llm_client.current_rail() == "a"
    assert llm_client.current_rail() == "unattributed"


def test_the_label_is_restored_even_when_the_rail_raises(isolated_stats):
    """A rail that dies mid-run must not leave its label smeared over everything that follows —
    in the long-lived MCP server that would mis-attribute every later call in the process."""
    with pytest.raises(RuntimeError):
        with llm_client.rail("frontier_execute"):
            raise RuntimeError("pull failed")
    assert llm_client.current_rail() == "unattributed"


def test_an_llm_call_is_attributed_too(isolated_stats):
    with llm_client.rail("oracle_refresh"):
        llm_client._bump_stats("classify", "m", 10, 5, 0.02)
    assert llm_client.spend_today_for_rail("oracle_refresh") == 0.02


def test_by_rail_absent_from_an_old_stats_file_does_not_crash(isolated_stats):
    """`_load_stats_once` merges by key, so a pre-existing api_stats.json with no `by_rail`
    must simply leave the fresh empty bucket in place. No migration."""
    llm_spend._stats_file().write_text(json.dumps({"lifetime": {"calls": 1, "input_tokens": 0,
                                                                "output_tokens": 0,
                                                                "cost_usd": 0.5}}))
    llm_spend._STATS_LOADED = False
    assert llm_client.spend_today_for_rail("anything") == 0.0
    assert llm_client.spend_total() == 0.5           # the old file still loads


# ── never destroy a recorded history ─────────────────────────────────────────────
#
# THE BUG THESE PIN (found 2026-08-16, reproduced on main): a single `pytest` run wiped
# ~/.opyt/api_stats.json — a $42.00 lifetime figure went to $0.00. Three pre-existing pieces
# composed into it. `atexit.register(flush_stats)` fires at process exit; the `isolated_stats`
# fixtures reset BOTH the path override to None AND `_STATS` to `_fresh_stats()` at teardown; so
# at exit `flush_stats` wrote an empty `_STATS` to the REAL path. Nothing failed, nothing was
# logged, and the only evidence was a spend history that had quietly become zero.
#
# Fixed at both levels, because they fail independently: production must never overwrite a file it
# has not read, and a test process must never resolve to the real store at all.


def test_flush_never_overwrites_a_file_it_has_not_read(isolated_stats):
    """THE PRODUCTION HALF. `_STATS_LOADED` False means this process has never merged the on-disk
    history into `_STATS`, so writing would drop whatever is there. It is also exactly the state
    a test fixture's teardown leaves behind.

    Safe to skip, always: every bump routes through `_load_stats_once` first, so an unloaded
    `_STATS` is by definition empty and the write would carry nothing."""
    isolated_stats.write_text(json.dumps({"lifetime": {"calls": 99, "input_tokens": 0,
                                                       "output_tokens": 0, "cost_usd": 42.0}}))
    llm_spend._STATS_LOADED = False
    llm_spend._STATS = llm_spend._fresh_stats()

    llm_client.flush_stats()

    assert json.loads(isolated_stats.read_text())["lifetime"]["cost_usd"] == 42.0


def test_flush_still_writes_once_the_history_has_been_read(isolated_stats):
    """The guard must not turn into a refusal to record. A spend loads first, so it writes."""
    llm_client.record_external_cost("openrouter-embed", 0.25)
    llm_client.flush_stats()
    assert json.loads(isolated_stats.read_text())["lifetime"]["cost_usd"] == 0.25


def test_flush_writes_a_first_file_on_a_fresh_install(isolated_stats):
    """No file yet is not the same as a file we have not read — a fresh install must still start
    recording. `_load_stats_once` marks itself loaded even when the file is absent."""
    assert not isolated_stats.exists()
    llm_client.record_external_cost("openrouter-embed", 0.01)
    llm_client.flush_stats()
    assert isolated_stats.exists()


def test_the_unset_override_resolves_into_a_sandbox_not_the_real_store():
    """THE TEST-PROCESS LAYER, on its own. `_override_stats_file_for_tests(None)` is what a
    fixture teardown leaves behind, and it used to mean "resolve to ~/.opyt". The session guard in
    tests/conftest.py rebinds `_stats_file` so the UNSET case lands in a sandbox instead — which
    is the hole, because the per-test override cannot cover the state after its own teardown."""
    from opyt_core.paths import opyt_path

    llm_client._override_stats_file_for_tests(None)
    assert llm_client._stats_file() != opyt_path("api_stats.json")


def test_a_pytest_run_leaves_the_real_stats_file_alone(tmp_path):
    """END TO END — this IS the reproduction that found the bug, kept as the regression test.

    A real `pytest` subprocess runs with `$OPYT_HOME` pointed at a sandbox holding a marker stats
    file; the marker must come back untouched. Verified non-vacuous by disabling each fix in turn:
    it passes with EITHER layer alone and fails only with BOTH off, which is the point — they are
    independent defences, and the production guard also protects processes that are not tests.

    ⚠️ It runs test_llm_client.py, NOT this file — this test lives here, and a subprocess running
    this file would spawn another subprocess, forever. That file is the right target anyway: it
    uses `isolated_stats` and so reproduces the exact teardown state that caused the wipe."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    home = tmp_path / "opyt-home"
    home.mkdir()
    marker = {"lifetime": {"calls": 7, "input_tokens": 0, "output_tokens": 0, "cost_usd": 42.0}}
    (home / "api_stats.json").write_text(json.dumps(marker))

    # ⚠️ $OPYT_CONFIG is pinned to the ACTIVE settings.yaml, derived at runtime — never a literal
    # path. `$OPYT_HOME` relocates the config too, and the packaged default does not declare the
    # roles these tests use, so without this the inner run dies on role resolution and the
    # assertion below would pass for the wrong reason.
    from opyt_core.config import config_path
    env = {**os.environ, "OPYT_HOME": str(home), "PYTHONPATH": str(repo),
           "OPYT_CONFIG": str(config_path())}
    run = subprocess.run([sys.executable, "-m", "pytest", "tests/test_llm_client.py",
                          "-q", "-p", "no:cacheprovider"],
                         cwd=repo, env=env, capture_output=True, text=True)
    assert run.returncode == 0, f"the inner run must pass, else this proves nothing:\n{run.stdout}"

    after = json.loads((home / "api_stats.json").read_text())
    assert after == marker, "a pytest run overwrote a real api_stats.json"


# ── the request that makes it available ──────────────────────────────────────────


def test_request_asks_for_usage_accounting(openrouter_reporting):
    """OpenRouter was observed returning usage.cost without this flag, but `call()` now depends on
    that field — the request states the dependency rather than relying on an undocumented default."""
    llm_client.call("content_quality", system="s", user="u")
    assert openrouter_reporting["body"]["usage"] == {"include": True}


def test_end_to_end_through_the_real_openrouter_path(openrouter_reporting, isolated_stats):
    """The unit tests above stub the backend; this one runs the real _call_openrouter_sync so the
    usage block is parsed from an actual response shape, not a hand-built dict."""
    resp = llm_client.call("content_quality", system="s", user="u")
    assert resp.cost_usd == pytest.approx(0.00118845)
    assert resp.input_tokens == 2503 and resp.output_tokens == 69
    assert (resp.raw or {}).get("provider") == "SambaNova", "served upstream must survive in raw"


# ── the static table: a miss must be LOUD, because a silent $0 uncaps a rail ────────

def test_an_unpriced_model_records_zero_but_says_so(monkeypatch):
    """`cost_for` returning 0.0 is correct (a wrong guess would be worse), but it must not be
    silent: `rail_budget_exhausted` reads this meter, so an unpriced model on the
    reported-cost-absent path hands a rail a ceiling that can never bind."""
    from pipeline import llm_spend

    logged = []
    monkeypatch.setattr(llm_spend, "log", lambda m: logged.append(m))
    monkeypatch.setattr(llm_spend, "_UNPRICED_LOGGED", set())

    assert llm_spend.cost_for("vendor/not-in-the-table", 1_000_000, 1_000_000) == 0.0
    llm_spend.cost_for("vendor/not-in-the-table", 5, 5)          # same model again
    assert len(logged) == 1 and "vendor/not-in-the-table" in logged[0]


def test_every_configured_role_model_is_priced():
    """The table is the BYOK path's only estimate — a role whose model is missing from it records
    $0 for every call. Regression: `deepseek-v4-flash` was a live role model with no row (found
    2026-08-28), alongside a llama row 5.5x low on input.

    EVERY role model, with no exemption list. The one model that ever needed one — a flat
    per-request web-search fee a token table cannot express — left with its role on 2026-08-28
    (docs/plans/2026-08-28-delete-the-cold-start-identity-anchor.md). Do not reintroduce an
    exemption set: it is how a merely-FORGOTTEN model hides."""
    from opyt_core.config import settings

    from pipeline import llm_spend

    roles = ((settings() or {}).get("llm_backends") or {}).get("roles") or {}
    missing = sorted({spec["model"] for spec in roles.values()
                      if (spec or {}).get("provider") == "openrouter" and spec.get("model")
                      and spec["model"] not in llm_spend._PRICING})
    assert missing == [], f"role models with no _PRICING row: {missing}"


def test_the_ocr_override_models_are_priced():
    """`call()` records under the model OVERRIDE, so the OCR cascade's spend looks up these keys
    and not the `vision` role's model. Both declared OCR candidates must resolve."""
    from pipeline import llm_spend, model_routing

    for m in model_routing.OCR_FALLBACKS:
        assert m in llm_spend._PRICING, f"{m} is a declared OCR model with no price row"
