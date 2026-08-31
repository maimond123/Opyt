"""
tests/test_credentials_registry.py

The five credential registries were collapsed into `opyt_core/credentials_registry.py` on
2026-08-13. This file is what makes that collapse provable rather than hopeful, in two halves:

  • CHARACTERIZATION — the derived views equal the pre-migration literals, spelled out here by
    hand. These are copied from the deleted code, so they are an INDEPENDENT statement of the
    old behavior, not a restatement of the new one. If a future edit to REGISTRY changes what
    the four surviving call sites see, these fail.

  • COVERAGE — every registry row reaches every surface a user meets: the `.env` scaffold, the
    preflight report, `opyt-keys --list`, and the validator dispatch. This is the half that the
    old design could not have: with five hand-maintained lists there was no `for` loop to write.

⚠️ THE POINT IS NOT THAT THE LISTS AGREE TODAY. They agreed on the morning this was written too,
by coincidence, after a month of not agreeing. The point is that agreement is now derived.
"""

from __future__ import annotations

import pathlib

import pytest

from opyt_core import credentials_registry as registry
from opyt_core import keys as core_keys
from pipeline import credentials, llm_client


# The exact literals that lived in the five hand-maintained lists, as of the commit before the
# collapse. Do NOT regenerate these from REGISTRY — that would make the assertions tautological.
#
# These are a FLOOR, not an inventory. The collapse must not have lost a row or remapped an
# alias, and that is what the assertions below check; a credential added afterwards for a new
# feature is not a regression of the collapse, so REGISTRY is allowed to be larger. What may
# never happen is one of these disappearing ACCIDENTALLY or pointing somewhere else.
#
# TWITTERAPI_KEY was the fourth, and it left the floor on 2026-08-30 — RETIRED, not lost. The
# provider it named was removed outright and every X read moved to the user's own browser session
# (docs/plans/2026-08-30-cut-x-ingestion-over-to-internal-graphql.md). Deleting the row is what
# the change was FOR, so re-adding it here would make this test refuse the deletion it should be
# indifferent to. The guard `retired-twitterapi-provider` is what stops it coming back.
PRE_MIGRATION_ENV_VARS = {"OPENROUTER_API_KEY", "GITHUB_TOKEN", "S2_API_KEY"}
PRE_MIGRATION_SERVICES = {
    "openrouter": "OPENROUTER_API_KEY",
    "github": "GITHUB_TOKEN",
    "semanticscholar": "S2_API_KEY",
}
PRE_MIGRATION_PROVIDER_ENV = {"openrouter": "OPENROUTER_API_KEY"}


# ── Characterization: the derived views match what they replaced ─────────────────


class TestDerivedViewsMatchThePreMigrationLiterals:
    def test_known_env_vars_are_unchanged(self):
        """Compared as a SET, because the order deliberately changed.

        Two hand-maintained lists of the same strings disagreed on order (`keys.py::KNOWN` led
        with the X provider's key, `create_env_template` with OPENROUTER_API_KEY), so one tuple
        cannot preserve both. REGISTRY is ordered required-first, which is what a user reading
        either surface wants. Membership is the contract; order is not.
        """
        assert PRE_MIGRATION_ENV_VARS <= set(registry.KNOWN)
        assert len(set(registry.KNOWN)) == len(registry.KNOWN)      # no duplicate rows

    def test_service_map_is_unchanged(self):
        """Every pre-collapse alias still resolves to the SAME env var. Checked as an items
        subset rather than dict equality so a later credential can be added without this test
        needing an edit — the property it protects is that none of these moved."""
        assert PRE_MIGRATION_SERVICES.items() <= registry.SERVICES.items()

    def test_provider_env_map_is_unchanged(self):
        """One entry since the direct Anthropic backend was retired 2026-08-13.

        Asserted as an exact dict on purpose: this fails if a provider is added to `_BACKENDS`
        without a registry row carrying its `provider` field."""
        assert registry.PROVIDER_ENV == PRE_MIGRATION_PROVIDER_ENV


class TestTheOldRegistriesAreNowTheSameObject:
    """Identity, not equality. Equality would still pass if someone rebuilt a private copy from
    REGISTRY at import time — which is a fifth list again, just one that happens to agree."""

    def test_keys_known_is_the_registry_view(self):
        assert core_keys.KNOWN is registry.KNOWN

    def test_credentials_services_is_the_registry_view(self):
        assert credentials.SERVICES is registry.SERVICES

    def test_llm_client_provider_env_is_the_registry_view(self):
        assert llm_client._PROVIDER_ENV is registry.PROVIDER_ENV


# ── Row well-formedness ──────────────────────────────────────────────────────────


class TestEveryRowIsUsable:
    def test_tiers_are_binary(self):
        """`required` or `optional` — no third value. A `dormant` tier was rejected when the
        Anthropic backend was deleted outright rather than carried across."""
        assert {c.tier for c in registry.REGISTRY} <= {"required", "optional"}

    def test_env_names_and_service_aliases_are_unique(self):
        envs = [c.env for c in registry.REGISTRY]
        services = [c.service for c in registry.REGISTRY]
        assert len(set(envs)) == len(envs)
        assert len(set(services)) == len(services)

    @pytest.mark.parametrize("cred", registry.REGISTRY, ids=lambda c: c.env)
    def test_a_user_can_act_on_a_missing_key(self, cred):
        """Every row carries what someone with a MISSING key needs: what breaks, and where to go
        get it. A row without a signup URL is a dead end at exactly the moment it is read."""
        assert cred.purpose, f"{cred.env} has no purpose"
        assert cred.signup_url.startswith("https://"), f"{cred.env} has no signup URL"

    def test_lookup_helpers_find_every_row_and_reject_strangers(self):
        for c in registry.REGISTRY:
            assert registry.by_service(c.service) is c
        assert registry.by_service("not-a-service") is None

    def test_the_registry_imports_nothing(self):
        """It is imported by `opyt_core/keys.py`, which must stay cheap and dependency-free, and
        it must never drag `requests` in behind the validators. `dataclasses` only."""
        src = pathlib.Path(registry.__file__).read_text()
        code_imports = [ln.strip() for ln in src.splitlines()
                        if ln.startswith(("import ", "from "))]
        assert code_imports == ["from __future__ import annotations",
                               "from dataclasses import dataclass"]


# ── Coverage: every row reaches every surface ────────────────────────────────────


class TestEveryRowReachesEverySurface:
    # Two preflight tests (every row lands in the right missing/warning bucket with its signup
    # link; a present key reads as ready) were deleted 2026-08-29 with their subject:
    # `lifecycle.check_prerequisites` had no production caller and went, per the
    # `retired-lifecycle-preflight` guard. The registry-derivation invariant they rode on is
    # enforced by `credential-registry-is-the-one-list`, not by any preflight.

    def test_key_list_reports_every_row_and_marks_the_tier(self, tmp_path, monkeypatch, capsys):
        """`opyt-keys --list` gains the tier, which is the first user-visible improvement that
        falls out of the collapse: a MISSING required key and a MISSING optional key used to look
        identical, so a working install looked broken."""
        monkeypatch.setenv("OPYT_HOME", str(tmp_path))
        for c in registry.REGISTRY:
            monkeypatch.delenv(c.env, raising=False)

        assert core_keys.main(["--list"]) == 0
        out = capsys.readouterr().out
        for c in registry.REGISTRY:
            assert c.env in out
            assert c.tier in out.split(c.env, 1)[1].splitlines()[0]

    def test_every_registry_service_has_a_validator(self, monkeypatch):
        """The one place the collapse deliberately STOPS. Validators call the network, so they
        stay in `pipeline/credentials.py` rather than moving into an import-free `opyt_core`
        module. That split is only safe if nothing falls through the dispatch — a service with no
        branch returns `True, "No validation available"`, which reports a DEAD key as fine.

        The concrete validators are stubbed, so this asserts routing and touches no network."""
        monkeypatch.setattr(llm_client, "validate_provider",
                            lambda provider, key: (True, f"routed:{provider}"))
        for name in ("_validate_github", "_validate_s2",
                     "_validate_opyt_service_token"):
            monkeypatch.setattr(credentials, name,
                                lambda key, _n=name: (True, f"routed:{_n}"))

        for c in registry.REGISTRY:
            ok, msg = credentials.validate_credential(c.service, "test-key")
            assert msg.startswith("routed:"), (
                f"{c.service} falls through validate_credential to 'No validation available' — "
                f"a dead key for it would be reported as valid")


# `_empty_config` WAS DELETED HERE 2026-08-15. It built a whole VaultConfig so the preflight
# had something to point its vault check at; both the check and the parameter are gone, so the
# helper had no remaining reason to exist.


# ── the vault check is retired ───────────────────────────────────────────────
# Its regression test (preflight takes no VaultConfig, never says "vault") went 2026-08-29 with
# the preflight itself — `lifecycle.check_prerequisites` was deleted caller-less. The property
# it pinned is now enforced structurally: the module is gone, and the `retired-vault-prerequisite`
# guard still bans the "Vault directory" string anywhere.
