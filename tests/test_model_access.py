"""
tests/test_model_access.py

One rule: with an OpenRouter key every model runs,
without one none does. Opyt's key pays for nothing. `credentials.model_access()` is the predicate,
re-read on every call so a key connected mid-session counts on the next one.
"""
from __future__ import annotations

import pytest
import yaml

from opyt_core import config
from pipeline import credentials, llm_client, model_routing


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A fresh home with no key anywhere: env cleared, home .env absent."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return tmp_path


@pytest.fixture()
def own_key(home, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-mine")


@pytest.fixture()
def backend(monkeypatch):
    """A counted fake OpenRouter backend."""
    calls: list = []

    def fake(model, system, user, max_tokens, **kw):
        calls.append(model)
        return "ok", 1, 1, 0.0, {}

    monkeypatch.setitem(llm_client._BACKENDS, "openrouter", fake)
    return calls


def _write_home_env(home, key: str) -> None:
    (home / ".env").write_text(f"OPENROUTER_API_KEY={key}\n")


def test_no_key_is_no_model_access(home):
    assert credentials.model_access() is False


def test_a_key_written_mid_process_counts_on_the_next_call(home):
    assert credentials.model_access() is False
    _write_home_env(home, "sk-or-mine")
    assert credentials.model_access() is True


def test_every_role_in_the_settings_is_one_the_code_calls():
    roles = yaml.safe_load(config._REPO_CONFIG.read_text())["llm_backends"]["roles"]
    assert set(roles) == set(model_routing.ROLES)


@pytest.mark.parametrize("role", model_routing.ROLES)
def test_with_no_key_every_role_is_refused_before_any_request(home, role):
    assert llm_client.preflight(role)


@pytest.mark.parametrize("role", model_routing.ROLES)
def test_every_role_runs_on_the_users_own_key(own_key, backend, role):
    assert llm_client.preflight(role) is None
    llm_client.call(role, system="s", user="u")
    assert len(backend) == 1


# ── image reads ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def ocr_gate(monkeypatch):
    """`chart_read` with a counted fake model call and the routing preflight skipped."""
    from pipeline import chart_read
    chart_read._reset_stage_for_tests()
    monkeypatch.setattr(chart_read, "_RESOLVED", "test-model")

    calls: list = []

    class _R:
        text = ("a transcription line of real substance, repeated for density " * 2
                + "\nTYPE: SCREENSHOT")
        output_tokens = 10

    monkeypatch.setattr(llm_client, "call", lambda **kw: calls.append(kw) or _R())
    return calls


def test_a_free_install_reads_no_images(ocr_gate, home):
    from pipeline import chart_read
    skipped_before = chart_read.stats_snapshot()["byok_only_skipped"]

    assert chart_read.read_image("https://pbs/0.jpg") is None   # no cache write anywhere
    assert ocr_gate == []                                       # zero model calls
    assert chart_read.stats_snapshot()["byok_only_skipped"] == skipped_before + 1


def test_a_key_connected_mid_session_reads_the_same_url(ocr_gate, home):
    """Nobody may memoize the predicate: the very next call after a home .env write spends."""
    from pipeline import chart_read
    assert chart_read.read_image("https://pbs/0.jpg") is None
    assert ocr_gate == []

    _write_home_env(home, "sk-or-mine")

    assert chart_read.read_image("https://pbs/0.jpg") is not None
    assert len(ocr_gate) == 1


def test_a_byok_user_can_turn_image_reads_off(ocr_gate, own_key, monkeypatch, tmp_path):
    """Their spend, their choice — `vision.enabled: false` in settings.yaml."""
    from pipeline import chart_read
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("vision:\n  enabled: false\n")
    monkeypatch.setenv("OPYT_CONFIG", str(cfg))

    assert chart_read.read_image("https://pbs/0.jpg") is None
    assert ocr_gate == []


def test_a_broken_settings_file_does_not_turn_off_a_paying_users_images(ocr_gate, own_key,
                                                                        monkeypatch, tmp_path):
    """Fail-safe direction: a missing or unreadable toggle reads as enabled."""
    from pipeline import chart_read
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(": not yaml : [")
    monkeypatch.setenv("OPYT_CONFIG", str(cfg))

    assert chart_read.read_image("https://pbs/0.jpg") is not None
    assert len(ocr_gate) == 1
