"""Config-reader unification (Workstream 1.2).

Every settings.yaml reader must resolve the ACTIVE config through the one
resolver — opyt_core.config.config_path() ($OPYT_CONFIG → ~/.opyt → repo
default) — instead of hardcoding REPO_ROOT/config/settings.yaml. Before this,
mcp_server.server, pipeline.ingestion.utils and pipeline.config each opened the
repo file directly, so a distributed install (whose real config lives in
~/.opyt) was ignored on those paths.

These tests are the guard: point $OPYT_CONFIG at a sentinel file and assert
each reader picks it up. StatePaths resolves runtime state under OPYT_HOME,
independently of the settings file's location, for both local and hosted users.
"""
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
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    return cfg


def test_resolver_itself_honors_opyt_config(sentinel_config):
    """The resolver honors $OPYT_CONFIG, proved by `config_path()` itself."""
    from opyt_core import config
    assert config.config_path() == sentinel_config


def test_pipeline_utils_load_yaml_config_honors_resolver(sentinel_config):
    from pipeline.ingestion import utils
    cfg = utils.load_yaml_config()
    assert cfg["vault"]["vault_path"] == SENTINEL_VAULT


def test_statepaths_resolve_honors_resolver(sentinel_config, monkeypatch):
    """resolve() reads no VALUE out of the file (no parsed field has a reader), so resolver-honoring shows only in the existence gate: it must
    succeed when $OPYT_CONFIG's file exists and raise when it doesn't."""
    from pipeline.config import ConfigNotFoundError, StatePaths
    StatePaths.resolve()                          # sentinel exists → constructs
    monkeypatch.setenv("OPYT_CONFIG", str(sentinel_config) + ".missing")
    with pytest.raises(ConfigNotFoundError):
        StatePaths.resolve()


def test_statepaths_resolve_uses_home_independently_of_settings(sentinel_config):
    from pipeline.config import StatePaths
    sp = StatePaths.resolve()
    assert sp.state_dir == sentinel_config.parent / "home" / "state"
    assert sp.state_file("trust_cache") == sp.state_dir / "trust_cache.json"


def test_statepaths_resolve_defaults_to_local_home(sentinel_config, tmp_path, monkeypatch):
    from opyt_core import paths
    from pipeline.config import StatePaths
    monkeypatch.delenv("OPYT_HOME", raising=False)
    monkeypatch.setattr(paths, "_DEFAULT_HOME", tmp_path / ".opyt")
    assert StatePaths.resolve().state_dir == tmp_path / ".opyt" / "state"


def test_state_paths_follows_home_changes(sentinel_config, tmp_path, monkeypatch):
    from pipeline.config import state_paths
    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    monkeypatch.setenv("OPYT_HOME", str(first_home))
    first = state_paths()
    monkeypatch.setenv("OPYT_HOME", str(second_home))
    second = state_paths()
    assert first.state_dir == first_home / "state"
    assert second.state_dir == second_home / "state"


def test_settings_reader_honors_resolver(sentinel_config):
    """A fresh install must read the USER's config, never the author's committed template.

    The property belongs to `opyt_core.config`, not to the MCP module that happened to call it.
    """
    from opyt_core import config
    assert config.settings()["sources"]["extraction_concurrency"] == 7


# ── Taxonomy resolver ───────────────────────────────────────────────────────
# There is no taxonomy file to resolve; the `retired-taxonomy-stack` guard bans its return.
