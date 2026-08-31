"""`TELEMETRY.md` is generated against the real DDL, not against intent.

The direction is the whole test: schema ⊆ doc. In an open-source project a telemetry document
that UNDERCOUNTS what the schema holds is worse than none — anyone can read the schema, find the
column the doc omitted, and quote both. So every table and every column in `service/store.py`'s
DDL has to appear in the document, and a column added later fails this test until it is written
down.

The reverse containment is deliberately NOT asserted. The doc says more than the schema does: it
also states what is never collected, which is exactly the part no column can carry.
"""
from __future__ import annotations

import re
from pathlib import Path

from service import store

TELEMETRY = Path(__file__).parents[2] / "TELEMETRY.md"


def test_every_table_in_the_ddl_is_documented():
    doc = TELEMETRY.read_text()
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", store._DDL)
    assert tables, "the DDL parse found no tables — the regex has gone stale, not the doc"
    for table in tables:
        assert f"`{table}`" in doc, f"{table} is in service.db and not in TELEMETRY.md"


def test_every_column_in_the_ddl_is_documented():
    """Matched as an inline-code span rather than as bare text, because `n` is a real column name
    and a bare-substring check for it passes on any English sentence. Column names in this DDL are
    lowercase and SQL keywords are not, which is what separates `zero_results` from `PRIMARY`."""
    doc = TELEMETRY.read_text()
    cols = {c for c in re.findall(r"^\s{2}(\w+)\s", store._DDL, re.M) if c.islower()}
    assert cols, "the DDL parse found no columns — the regex has gone stale, not the doc"
    for col in cols:
        assert f"`{col}`" in doc, f"{col} is a column in service.db and not in TELEMETRY.md"


def test_the_never_collected_list_survives():
    """The five refusals, each pinned by the term the ruling names it with. A doc that quietly
    dropped one would still pass the schema check above, because a refusal has no column."""
    doc = TELEMETRY.read_text().lower()
    for refusal in ("query text", "ip address", "atom", "timestamp", "client"):
        assert refusal in doc


def test_redeeming_a_grant_discloses_what_is_counted(svc):
    """The disclosure arrives BEFORE the first query, which is the only moment it is worth
    anything. Asserted on the presence and on the three refusals it has to name — not on the
    wording, which is free to be reworded."""
    code = svc.client.post("/v1/grant", json={}, headers=svc.owner_hdr).json()["code"]
    notice = svc.client.post("/v1/redeem", json={"code": code}).json()["notice"]
    low = notice.lower()
    assert "query text" in low and "ip address" in low and "atoms you read" in low
    assert "TELEMETRY.md" in notice
