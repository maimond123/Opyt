from pipeline.oracles.sources.threads.walk import Walk
from pipeline.oracles.sources.threads.walk import advance
from pipeline.oracles.sources.threads.walk import restore


def post(tid, parent=None, *, replies=1, author="a", conversation="100"):
    return {"id": str(tid), "author": {"id": author}, "isReply": parent is not None,
            "inReplyToUserId": author, "inReplyToStatusId": str(parent or ""),
            "conversationId": conversation, "replyCount": replies}


def test_a_long_thread_resumes_between_requests_and_exceeds_the_old_five_request_cap():
    walk, calls = Walk(), []
    for focal in range(100, 108):
        assert walk.next_focal("100") == str(focal)
        calls.append(str(focal))
        page = [post(focal, focal - 1 if focal > 100 else None)]
        if focal < 107:
            page.append(post(focal + 1, focal))
        walk = restore(advance(walk, "100", "a", str(focal), page).saved())
    assert walk.next_focal("100") is None
    assert len(calls) == len(set(calls)) == 8
    assert len(walk.tweets) == 8


def test_a_repeated_page_exhausts_the_walk_instead_of_spinning():
    page = [post(100), post(101, 100)]
    walk = advance(Walk(), "100", "a", "100", page)
    walk = advance(walk, "100", "a", "101", page)
    assert walk.next_focal("100") is None


def test_other_authors_other_conversations_and_replies_to_commenters_are_excluded():
    to_commenter = {**post(103, 100), "inReplyToUserId": "someone_else"}
    walk = advance(Walk(), "100", "a", "100", [
        post(100), post(101, 100, author="b"), post(102, 100, conversation="99"),
        to_commenter, post(104, 100)])
    assert [t["id"] for t in walk.tweets] == ["100", "104"]
    assert walk.next_focal("100") == "104"


def test_only_the_root_request_can_establish_that_a_thread_is_gone():
    assert advance(Walk(), "100", "a", "100", []).gone
    walk = advance(Walk(), "100", "a", "100", [post(100), post(101, 100)])
    walk = advance(walk, "100", "a", "101", [])
    assert not walk.gone
    assert walk.next_focal("100") is None
