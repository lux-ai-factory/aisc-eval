"""A helper model's address is checked against the project's network rule before the plugin starts: an
address the rule refuses, or a name that does not resolve, fails the run naming why, and no environment is
built and no plugin started."""

import socket

import pytest

from test_helper_resolution import RESOLVED, _helper_input, run  # noqa: F401  (fixtures)
from test_plugin_process import _Answer, platform, worker  # noqa: F401  (fixtures)


@pytest.fixture
def names(monkeypatch):
    """Name resolution from a table: llm.example is a public address, anything else does not resolve."""
    table = {"llm.example": "93.184.216.34"}

    def fake(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror(f"{host} does not resolve")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return table


def test_a_helper_endpoint_the_rule_refuses_fails_the_run_before_the_plugin(run, names):  # noqa: F811
    run["resolve"] = _Answer(
        200,
        {
            **RESOLVED,
            "endpoint_url": "http://169.254.169.254/v1",
            "allowed_hosts": [],
            "denied_addresses": ["169.254.169.254"],
            "floor_hosts": [],
        },
    )
    with pytest.raises(RuntimeError, match="never"):
        run["go"]([_helper_input()])
    assert run["worker"]["popen"] == [] and run["venvs"] == []


def test_a_helper_endpoint_on_an_internal_address_fails_the_run(run, names):  # noqa: F811
    names["judge.internal"] = "10.0.0.5"
    run["resolve"] = _Answer(
        200,
        {
            **RESOLVED,
            "endpoint_url": "http://judge.internal/v1",
            "allowed_hosts": [],
            "denied_addresses": [],
            "floor_hosts": [],
        },
    )
    with pytest.raises(RuntimeError, match="internal address"):
        run["go"]([_helper_input()])
    assert run["worker"]["popen"] == []


def test_a_helper_endpoint_that_does_not_resolve_fails_the_run_naming_it(run, names):  # noqa: F811
    run["resolve"] = _Answer(
        200,
        {
            **RESOLVED,
            "endpoint_url": "https://gone.example/v1",
            "allowed_hosts": [],
            "denied_addresses": [],
            "floor_hosts": [],
        },
    )
    with pytest.raises(RuntimeError, match="gone.example does not resolve"):
        run["go"]([_helper_input()])
    assert run["worker"]["popen"] == []
