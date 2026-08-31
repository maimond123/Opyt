"""record_affiliation — a footprint source the single-author gate SKIPPED as multi-author/org is
kept as an AFFILIATION (an `org:` fact-node + an attested `affiliated_with` edge from the person),
NEVER as content. No atoms are ever minted; the recording is idempotent on the org host; and a
thin input (no Oracle id / no resolvable org host) records nothing rather than a dangling edge."""
from __future__ import annotations

import pytest

from pipeline.kb import derive, eligibility, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def test_org_entity_id():
    # org:{canonical_host}, path dropped; empty/garbage → the stable never-merging singleton.
    assert derive.org_entity_id("https://anthropic.com/research") == "org:anthropic.com"
    assert derive.org_entity_id("") == "org:unknown"


def test_record_affiliation_creates_the_org_node(conn):
    """The gate SKIPPING an org must not DISCARD it. The person→org relation used to ride an
    `affiliated_with` edge; that table went 2026-08-23 with no reader, so the org node is the
    whole record now. The `org:` id PREFIX is the marker — `entities.kind` was deleted
    2026-08-23 for having no reader, so an assertion on it would test a field nothing consults."""
    schema.upsert_entity(conn, "x:user:7", name="Chris")
    org_id = eligibility.record_affiliation(conn, "x:user:7", "https://anthropic.com",
                                            org_name="Anthropic")
    assert org_id == "org:anthropic.com"
    org = schema.get_entity(conn, org_id)
    assert org["name"] == "Anthropic"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0    # a fact, not content


def test_record_affiliation_idempotent_on_host(conn):
    schema.upsert_entity(conn, "x:user:7")
    eligibility.record_affiliation(conn, "x:user:7", "https://anthropic.com")
    eligibility.record_affiliation(conn, "x:user:7", "https://anthropic.com/careers")   # same host
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id LIKE 'org:%'").fetchone()[0] == 1


def test_record_affiliation_guards_thin_inputs(conn):
    assert eligibility.record_affiliation(conn, "", "https://anthropic.com") is None     # no person
    assert eligibility.record_affiliation(conn, "x:user:7", "") is None                  # no org host
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id LIKE 'org:%'").fetchone()[0] == 0
