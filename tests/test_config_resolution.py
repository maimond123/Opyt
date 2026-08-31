"""Config-reader unification (Workstream 1.2).

Every settings.yaml reader must resolve the ACTIVE config through the one
resolver — opyt_core.config.config_path() ($OPYT_CONFIG → ~/.opyt → repo
default) — instead of hardcoding REPO_ROOT/config/settings.yaml. Before this,
mcp_server.server, pipeline.ingestion.utils and pipeline.config each opened the
repo file directly, so a distributed install (whose real config lives in
~/.opyt) was ignored on those paths.

These tests are the guard: point $OPYT_CONFIG at a sentinel file and assert
each reader picks it up. They also pin the invariant that StatePaths.resolve
keeps state_dir anchored to the REPO tree, not to wherever the settings file
resolved from — otherwise routing through the resolver would silently relocate
state_dir to ~/.

(That last invariant was originally justified by taxonomy loading. The taxonomy
stack was deleted 2026-08-13, but the invariant outlives its first motivation:
state_dir still holds real state files, and relocating it on a config move would
still be wrong.)
"""
from pathlib import Path

import pytest
import yaml

SENTINEL_VAULT = "/tmp/opyt-sentinel-vault-zzz"


@pytest.fixture
def sentinel_config(tmp_path, monkeypatch):
    """Write a sentinel settings.yaml and make it the active config via $OPYT_CONFIG."""
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(yaml.safe_dump({
        "vault": {"vault_path": SENTINEL_VAULT, "raw_path": SENTINEL_VAULT + "/raw"},
        "sources": {"extraction_concurrency": 7},
    }))
    monkeypatch.setenv("OPYT_CONFIG", str(cfg))
    return cfg


def test_resolver_itself_honors_opyt_config(sentinel_config):
    """`config.vault_path()` WAS asserted here too, until it was deleted 2026-08-13 with the
    vault. What it proved — that the resolver honors $OPYT_CONFIG — is proved by `config_path()`
    itself, so nothing was lost by dropping the second assert rather than repointing it."""
    from opyt_core import config
    assert config.config_path() == sentinel_config


def test_pipeline_utils_load_yaml_config_honors_resolver(sentinel_config):
    from pipeline.ingestion import utils
    cfg = utils.load_yaml_config()
    assert cfg["vault"]["vault_path"] == SENTINEL_VAULT


def test_statepaths_resolve_honors_resolver(sentinel_config, monkeypatch):
    """resolve() reads no VALUE out of the file (every parsed field was deleted 2026-08-29
    as reader-less), so resolver-honoring shows only in the existence gate: it must
    succeed when $OPYT_CONFIG's file exists and raise when it doesn't."""
    from pipeline.config import ConfigNotFoundError, StatePaths
    StatePaths.resolve()                          # sentinel exists → constructs
    monkeypatch.setenv("OPYT_CONFIG", str(sentinel_config) + ".missing")
    with pytest.raises(ConfigNotFoundError):
        StatePaths.resolve()


def test_statepaths_resolve_keeps_state_dir_repo_anchored(sentinel_config):
    """state_dir must NOT follow the config file into ~/.opyt."""
    import pipeline.config as pc
    sp = pc.StatePaths.resolve()
    repo_root = Path(pc.__file__).resolve().parent.parent
    assert sp.state_dir == repo_root / "state"


def test_settings_reader_honors_resolver(sentinel_config):
    """A fresh install must read the USER's config, never the author's committed template.

    This was `test_mcp_server_load_config_honors_resolver` until 2026-08-13, asserting through
    `mcp_server.server._load_config()`. That function was caller-free and existed only to be
    asserted on, so it was deleted and the test repointed at the resolver it was really testing.
    The property belongs to `opyt_core.config`, not to the MCP module that happened to call it.
    """
    from opyt_core import config
    assert config.settings()["sources"]["extraction_concurrency"] == 7


# ── Taxonomy resolver — ALL FOUR TESTS DELETED 2026-08-13 ───────────────────
# They covered `taxonomy_path` / `taxonomy_write_path` / `$OPYT_TAXONOMY` and the read-vs-write
# split, every one of which was deleted with the taxonomy stack. Atoms carried `about_topics`
# directly at the time (itself retired 2026-08-17 — see `retired-about-topics-column`); there
# is no taxonomy file to resolve either way. See the `retired-taxonomy-stack` guard.
#
# One of them is worth naming, because it is the kind of test that looks load-bearing:
# `test_pipeline_readers_honor_taxonomy_resolver` existed because `load_taxonomy` was implemented
# THREE times and the copies could drift. The copies were collapsed in 2026-08-07, which made the
# drift impossible rather than merely detected — so the test was already guarding a closed class of
# bug before its subject was deleted.


# ── cookies.profile — the other half of cookies.browser ─────────────────────
#
# `cookie_browser()` resolved through settings.yaml while the PROFILE resolved through
# $X_CHROME_PROFILE only. Two halves of one choice, persisted in two different places, and
# only one of them survived a reboot. `onboard` needs somewhere durable to write the user's
# pick, so `cookie_profile()` mirrors `cookie_browser()` exactly.

def _settings_with(tmp_path, monkeypatch, cookies: dict):
    """Write a settings.yaml carrying a `cookies:` block and make it the active config.
    Mirrors the `sentinel_config` fixture above; parameterized because these tests vary
    the block under test."""
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(yaml.safe_dump({"vault": {"vault_path": "/tmp/v"}, "cookies": cookies}))
    monkeypatch.setenv("OPYT_CONFIG", str(cfg))
    return cfg


def test_cookie_profile_reads_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("X_CHROME_PROFILE", raising=False)
    _settings_with(tmp_path, monkeypatch, {"browser": "chrome", "profile": "Profile 1"})
    from opyt_core.config import cookie_profile
    assert cookie_profile() == "Profile 1"


def test_cookie_profile_env_overrides_settings(monkeypatch, tmp_path):
    _settings_with(tmp_path, monkeypatch, {"profile": "Profile 1"})
    monkeypatch.setenv("X_CHROME_PROFILE", "Profile 2")
    from opyt_core.config import cookie_profile
    assert cookie_profile() == "Profile 2"


def test_cookie_profile_absent_is_none(monkeypatch, tmp_path):
    monkeypatch.delenv("X_CHROME_PROFILE", raising=False)
    _settings_with(tmp_path, monkeypatch, {"browser": "auto"})
    from opyt_core.config import cookie_profile
    assert cookie_profile() is None
