from pipeline.platforms.x.contracts import XTransport
from pipeline.platforms.x.local import LocalXTransport, auth_headers


def test_the_local_transport_implements_the_x_transport_contract():
    cookies = {"auth_token": "token", "ct0": "csrf", "twid": "u%3D42"}
    local = LocalXTransport(cookies, auth_headers(cookies, "https://x.com/home"))

    assert isinstance(local, XTransport)
