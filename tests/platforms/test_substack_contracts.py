from pipeline.platforms.substack.local import LocalFollows, LocalSaved, LocalSubscriptionList
from pipeline.platforms.substack.contracts import (
    FollowSource,
    SavedSource,
    SubscriptionSource,
)


def test_the_local_readers_implement_their_contracts():
    assert isinstance(LocalFollows(), FollowSource)
    assert isinstance(LocalSubscriptionList(), SubscriptionSource)
    assert isinstance(LocalSaved(), SavedSource)
