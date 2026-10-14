"""The one HTTP transport every web read passes through: it asks robots.txt before each read
(`robots.check`), and bounds concurrent reads of each host, including redirects and secondary
fetches.

The web executor supplies a shared gate for its workers, and inside it every redirect hop is a
read of its own, asked and gated. Outside that scope existing HTTP callers keep their transport
behavior: the requested URL is asked, and the transport follows redirects itself. A redirect
releases its old host before acquiring the next, so two sites redirecting to each other cannot
deadlock.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from urllib.parse import urljoin, urlsplit

import feedparser
import requests

from pipeline import robots

_CURRENT = ContextVar("web_host_access", default=None)


class Hosts:
    def __init__(self):
        self._guard = Lock()
        self._locks = {}

    def lock(self, url):
        host = urlsplit(url).hostname or ""
        with self._guard:
            return self._locks.setdefault(host.lower(), Lock())


@contextmanager
def scope(hosts):
    token = _CURRENT.set(hosts)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _user_agent(headers) -> str:
    """The User-Agent this read sends: its own header, else the transport's default."""
    for key, value in (headers or {}).items():
        if key.lower() == "user-agent":
            return value
    return requests.utils.default_user_agent()


def request(send, url, **kwargs):
    """Run a GET/HEAD transport under the current gate, following redirects one host at a time.
    Raises `robots.Refused` (a `requests.RequestException`) for a URL robots.txt closes."""
    hosts = _CURRENT.get()
    if hosts is None:
        robots.check(url, _user_agent(kwargs.get("headers")))
        return send(url, **kwargs)
    options = dict(kwargs)
    follow = options.pop("allow_redirects", True)
    for _ in range(31):
        lock = hosts.lock(url)
        lock.acquire()
        try:
            robots.check(url, _user_agent(options.get("headers")))
            response = send(url, **options, allow_redirects=False)
        except BaseException:
            lock.release()
            raise
        if follow and response.status_code in (301, 302, 303, 307, 308) and response.headers.get("Location"):
            destination = urljoin(url, response.headers["Location"])
            try:
                response.close()
            finally:
                lock.release()
            options.pop("params", None)
            if urlsplit(destination).netloc != urlsplit(url).netloc:
                options.pop("cookies", None)
                options.pop("auth", None)
                options["headers"] = {k: v for k, v in options.get("headers", {}).items()
                                      if k.lower() not in ("authorization", "cookie", "host")}
            url = destination
            continue
        if options.get("stream"):
            # Streaming callers close the response after their bounded probe/download.
            close = response.close
            released = False

            def release():
                nonlocal released
                try:
                    return close()
                finally:
                    if not released:
                        released = True
                        lock.release()
            response.close = release
        else:
            lock.release()  # non-streaming transports read the body before returning
        return response
    raise RuntimeError("too many HTTP redirects")


def feed_input(url):
    """Feedparser receives bytes in managed work, so its private URL opener cannot bypass the gate.
    Outside it feedparser fetches the URL itself, as itself, once robots.txt allows that."""
    if _CURRENT.get() is None:
        robots.check(url, feedparser.USER_AGENT)
        return url
    response = request(requests.get, url, timeout=20)
    response.raise_for_status()
    return response.content
