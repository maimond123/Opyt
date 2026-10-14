"""Walk one-page reads to the end by the same rule a durable walk applies (`continuation`), for
tests that pin how a page read ends its collection."""
from pipeline.platforms.collection_page import continuation


def walk(read) -> list[dict]:
    """Every distinct record, in order, from `read(cursor)` until the collection ends."""
    cursor, seen, cursors, keys, out = None, set(), set(), set(), []
    while True:
        page = read(cursor)
        following = continuation(page, seen, cursors)
        for key, record in page.records:
            if key not in keys:
                keys.add(key)
                out.append(record)
        seen.update(page.observed)
        if following is None:
            return out
        cursors.add(following)
        cursor = following
