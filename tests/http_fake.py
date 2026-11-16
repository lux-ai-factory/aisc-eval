"""The worker's HTTP, faked where it is made: every call of the service client (aisc_service_client) goes to
`verbs[method](url, **kwargs)` instead of the network. A fake raises `unreachable()` for a backend or platform
that cannot be reached, and its answers need only what the worker reads (status_code, json(), text, content)."""

import httpx


def route(monkeypatch, **verbs):
    """Each call goes to verbs[get|put|patch|post|delete](url, headers=..., timeout=..., json=..., ...). A second
    route in the same test adds its verbs to the first's, replacing those it names."""
    table = {**getattr(httpx.Client.request, "verbs", {}), **verbs}

    def request(self, method, url, **kwargs):
        return table[method.lower()](str(url), **kwargs)

    request.verbs = table
    monkeypatch.setattr(httpx.Client, "request", request)


def unreachable(message: str = "connection refused") -> httpx.ConnectError:
    """What a fake raises when nothing answers."""
    return httpx.ConnectError(message)


def timeout_of(pair) -> httpx.Timeout:
    """The httpx timeout a (connect, read) pair of seconds is sent as."""
    return httpx.Timeout(pair[1], connect=pair[0])
