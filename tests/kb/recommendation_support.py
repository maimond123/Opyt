"""Drive the real resumable recommendation source in transport-fixture tests."""
import json

from pipeline.curation import recommended as recs
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop.source import Context
from pipeline.oracles.sources.web.recommendations import Recommendations
from pipeline.oracles.sources import scraper_tables


def run_pass(conn, now=None):
    scraper_tables.init_schema(conn)
    now = now or utc_now()
    source = Recommendations()
    status = "not_due"
    for _ in range(500):
        unit = source.next_unit(conn, now)
        if unit is None:
            break
        status = source.record(conn, unit, source.fetch(conn, unit, Context("web"))).status
    else:
        raise AssertionError("recommendations did not finish")
    if status == "not_due":
        return {"status": status}
    results = [{**dict(r), "recommendations": json.loads(r["payload"] or "[]")}
               for r in conn.execute("SELECT * FROM web_recommendation_reads")]
    out = recs.store_recommendations(conn, results)
    return {"status": status, **out}
