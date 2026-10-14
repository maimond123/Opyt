import pytest

from pipeline.platforms.substack.account import preload_user


@pytest.mark.parametrize(
    ("decoded_shape", "expected"),
    [
        ({"user": {"id": 42}}, 42),
        ({"user": {"id": "42"}}, "42"),
        ({"user": {"id": 0}}, 0),
        ({"user": {}}, None),
        ({}, None),
    ],
)
def test_preload_user_preserves_supported_wrapper_and_id_values(decoded_shape, expected):
    assert preload_user(decoded_shape) == expected


@pytest.mark.parametrize("decoded_shape", [[{"user": {"id": 1}}], "x", None, {"user": "x"},
                                           {"user": [1]}])
def test_preload_user_refuses_preloads_that_are_not_objects(decoded_shape):
    """B13: these leaked AttributeError (or read as signed out)."""
    from pipeline.platforms.substack.contracts import SubstackListingError
    with pytest.raises(SubstackListingError):
        preload_user(decoded_shape)
