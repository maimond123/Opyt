"""
tests/test_credentials_validation.py

`pipeline/credentials.py` stays vendor-agnostic: the hardcoded Anthropic validator and SDK are
gone, and nothing brings them back.
"""

from __future__ import annotations

import pathlib

from pipeline import credentials


# ── credentials routing ──────────────────────────────────────────────────────


class TestValidateCredentialAgnostic:
    def test_vendor_specific_validator_is_gone(self):
        # The hardcoded Anthropic validator was deleted under the OpenRouter migration.
        assert not hasattr(credentials, "_validate_anthropic")

    def test_no_direct_anthropic_or_haiku_in_source(self):
        # Migration completeness: credentials.py must not resurrect the SDK/model ref.
        src = pathlib.Path(credentials.__file__).read_text()
        assert "claude-haiku" not in src
        assert "anthropic.Anthropic" not in src
        assert "import anthropic" not in src
