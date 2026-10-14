"""`invite.parse` and `invite.link` — the invite format as plain values, one test per row of its
table."""
from __future__ import annotations

import pytest

from pipeline.sharing import account, invite

CODE = "aB3" * 14 + "z"          # 43 letters and digits
CONFIGURED = "https://configured.test"


def test_a_bare_code_redeems_at_the_configured_service():
    assert invite.parse(CODE, CONFIGURED) == invite.Found(CODE, CONFIGURED)


def test_a_public_link_redeems_at_the_hosted_service():
    found = invite.parse(f"https://useopyt.com/invite#{CODE}", CONFIGURED)
    assert found == invite.Found(CODE, account.DEFAULT_SERVICE_URL)


def test_a_self_hosted_link_redeems_at_its_own_origin_port_included():
    found = invite.parse(f"https://self-hosted.test:8443/invite#{CODE}", CONFIGURED)
    assert found == invite.Found(CODE, "https://self-hosted.test:8443")


def test_a_lookalike_host_is_not_the_public_one():
    found = invite.parse(f"https://notuseopyt.com/invite#{CODE}", CONFIGURED)
    assert found == invite.Found(CODE, "https://notuseopyt.com")


@pytest.mark.parametrize("pasted", ["", "no code here", CODE[:-1], CODE + "x", "-" + CODE[:-1]])
def test_no_complete_code_is_no_invite(pasted):
    assert invite.parse(pasted, CONFIGURED) is None


def test_the_link_carries_the_code_in_its_fragment():
    assert invite.link(CODE) == f"https://useopyt.com/invite#{CODE}"
    assert invite.parse(invite.link(CODE), CONFIGURED).code == CODE


def test_every_minted_alphabet_character_is_one_the_pattern_finds():
    assert invite.parse(invite.CODE_ALPHABET[:invite.CODE_LEN], CONFIGURED) is not None
