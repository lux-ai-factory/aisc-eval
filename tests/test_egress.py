"""A plugin test reaches the network only through the worker's egress proxy, and the proxy admits only the
hosts of that test: the platform's facade, its helper endpoints and the operator's extra hosts, each on an
address the network rule allows. Anything else, a public host included, is refused with a reason. The firewall
lets the run users reach the proxy and nothing else, IPv6 and DNS included. The run's proxy token is scrubbed
from everything the run uploads, and a root worker without the firewall refuses to start a plugin."""

import base64
import contextlib
import hashlib
import http.server
import importlib
import ipaddress
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

from aisc_eval import celery_tasks
from test_plugin_process import ONE_MEASURE, _run, worker  # noqa: F401  (fixtures)

TOKEN = "fake-run-proxy-token"


def egress():
    try:
        return importlib.import_module("aisc_eval.egress")
    except ImportError:
        pytest.fail(
            "there is no egress policy (aisc_eval.egress): a plugin reaches any host"
        )


def egress_proxy():
    try:
        return importlib.import_module("aisc_eval.egress_proxy")
    except ImportError:
        pytest.fail(
            "there is no egress proxy (aisc_eval.egress_proxy): a plugin reaches any host"
        )


def resolving(monkeypatch, table):
    def fake(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


def policy(e):
    return e.RunEgress(
        frozenset({"platform:8000", "api.example"}),
        {
            "allowed_hosts": ["platform:8000"],
            "denied_addresses": ["172.20.0.9"],
            "floor_hosts": ["platform:8000"],
        },
    )


# the policy


def test_a_host_of_the_run_is_reached_on_its_checked_address(monkeypatch):
    e = egress()
    resolving(monkeypatch, {"api.example": "93.184.216.34"})
    assert e.decide("api.example", 443, policy(e)) == ipaddress.ip_address(
        "93.184.216.34"
    )


def test_any_other_host_is_refused_even_a_public_one(monkeypatch):
    e = egress()
    resolving(monkeypatch, {"pypi.org": "151.101.0.223"})
    with pytest.raises(e.Refused, match="not an address this test may reach"):
        e.decide("pypi.org", 443, policy(e))


@pytest.mark.parametrize(
    "name,ip",
    [
        ("platform", "172.20.0.9"),
        ("api.example", "169.254.169.254"),
        ("api.example", "10.0.0.7"),
    ],
)
def test_a_run_host_pointing_at_a_service_metadata_or_internal_address_is_refused(
    monkeypatch, name, ip
):
    e = egress()
    resolving(monkeypatch, {name: ip})
    with pytest.raises(e.Refused):
        e.decide(name, 8000 if name == "platform" else 443, policy(e))


def test_the_firewall_lets_the_pool_reach_only_the_proxy():
    e = egress()
    assert e.firewall_rules(range(10100, 10200), 3128) == [
        [
            "iptables",
            "-A",
            "OUTPUT",
            "-m",
            "owner",
            "--uid-owner",
            "10100-10199",
            "-d",
            "127.0.0.1",
            "-p",
            "tcp",
            "--dport",
            "3128",
            "-j",
            "ACCEPT",
        ],
        [
            "iptables",
            "-A",
            "OUTPUT",
            "-m",
            "owner",
            "--uid-owner",
            "10100-10199",
            "-j",
            "REJECT",
        ],
        [
            "ip6tables",
            "-A",
            "OUTPUT",
            "-m",
            "owner",
            "--uid-owner",
            "10100-10199",
            "-j",
            "REJECT",
        ],
    ]


def test_the_firewall_is_in_force_only_when_every_rule_is_there():
    e = egress()
    asked = []

    class _Answer:
        def __init__(self, code):
            self.returncode = code

    def present(argv, **kw):
        asked.append(argv)
        return _Answer(0)

    assert e.in_force(run=present) is True
    assert all(argv[1] == "-C" for argv in asked) and len(asked) == 3
    assert (
        e.in_force(run=lambda argv, **kw: _Answer(1 if argv[0] == "ip6tables" else 0))
        is False
    )


def test_a_rule_file_is_readable_only_by_the_proxy_and_gone_after_the_run(
    tmp_path, monkeypatch
):
    e = egress()
    monkeypatch.setattr(os, "chown", lambda *a: None)
    path = e.register(TOKEN, policy(e), tmp_path)
    assert path.name == hashlib.sha256(TOKEN.encode()).hexdigest() + ".json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert TOKEN not in path.read_text()
    e.unregister(TOKEN, tmp_path)
    assert not path.exists()


def test_the_plugins_proxy_variables_name_the_proxy_with_the_runs_token():
    e = egress()
    env = e.proxy_env(TOKEN)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env[name] == f"http://run:{TOKEN}@127.0.0.1:3128"
    assert env["NO_PROXY"] == env["no_proxy"] == ""


# the proxy, in this process on a free port, in front of a stub server on loopback


@pytest.fixture
def upstream():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append({"path": self.path, "headers": dict(self.headers)})
            body = b"hello from upstream"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.seen = seen
    yield server
    server.shutdown()
    server.server_close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def proxy(tmp_path, upstream, monkeypatch):
    """The proxy's port and the upstream's host:port; None while there is no proxy (each test then fails)."""
    try:
        e = importlib.import_module("aisc_eval.egress")
        p = importlib.import_module("aisc_eval.egress_proxy")
    except ImportError:
        return None
    monkeypatch.setattr(os, "chown", lambda *a: None)
    target = f"127.0.0.1:{upstream.server_address[1]}"
    e.register(
        TOKEN,
        e.RunEgress(
            frozenset({target}),
            {
                "allowed_hosts": [target],
                "denied_addresses": [],
                "floor_hosts": [target],
            },
        ),
        tmp_path,
    )
    port = _free_port()
    threading.Thread(
        target=p.serve,
        kwargs={"host": "127.0.0.1", "port": port, "policies": tmp_path},
        daemon=True,
    ).start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with (
            contextlib.suppress(OSError),
            socket.create_connection(("127.0.0.1", port), 0.2),
        ):
            break
        time.sleep(0.05)
    return {"port": port, "target": target}


def _auth(token=TOKEN):
    return "Basic " + base64.b64encode(f"run:{token}".encode()).decode()


def running(proxy):
    if proxy is None:
        pytest.fail(
            "there is no egress proxy (aisc_eval.egress_proxy): a plugin reaches any host"
        )
    return proxy


def _connect(proxy, target, auth=True):
    s = socket.create_connection(("127.0.0.1", proxy["port"]), 5)
    head = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n"
    if auth:
        head += f"Proxy-Authorization: {_auth() if auth is True else auth}\r\n"
    s.sendall((head + "\r\n").encode())
    answer = b""
    while b"\r\n\r\n" not in answer:
        chunk = s.recv(4096)
        if not chunk:
            break
        answer += chunk
    return s, answer.split(b"\r\n", 1)[0].decode()


def test_connect_to_a_run_host_is_tunnelled(proxy):
    running(proxy)
    s, status = _connect(proxy, proxy["target"])
    with s:
        assert " 200 " in status, status
        s.sendall(b"GET /through HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        body = b""
        while chunk := s.recv(4096):
            body += chunk
    assert b"hello from upstream" in body


def test_connect_elsewhere_is_403_with_the_reason(proxy):
    running(proxy)
    s, status = _connect(proxy, "pypi.org:443")
    s.close()
    assert " 403 " in status, status


@pytest.mark.parametrize("auth", [False, _auth("not-a-run-token")])
def test_no_token_or_an_unknown_one_is_407(proxy, auth):
    running(proxy)
    s, status = _connect(proxy, proxy["target"], auth=auth)
    s.close()
    assert " 407 " in status, status


def test_a_plain_request_is_forwarded_without_forwarded_headers(proxy, upstream):
    running(proxy)
    with socket.create_connection(("127.0.0.1", proxy["port"]), 5) as s:
        s.sendall(
            (
                f"GET http://{proxy['target']}/plain HTTP/1.1\r\nHost: {proxy['target']}\r\n"
                f"Proxy-Authorization: {_auth()}\r\nConnection: close\r\n\r\n"
            ).encode()
        )
        answer = b""
        while chunk := s.recv(4096):
            answer += chunk
    assert b"hello from upstream" in answer
    [seen] = upstream.seen
    names = {k.lower() for k in seen["headers"]}
    assert not names & {"x-forwarded-for", "forwarded", "proxy-authorization"}, names


def test_a_tunnel_carries_five_megabytes_each_way_intact(proxy, tmp_path, monkeypatch):
    running(proxy)
    e = egress()
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)
    target = f"127.0.0.1:{echo.getsockname()[1]}"
    e.register(
        TOKEN,
        e.RunEgress(
            frozenset({target}),
            {
                "allowed_hosts": [target],
                "denied_addresses": [],
                "floor_hosts": [target],
            },
        ),
        tmp_path,
    )

    def serve_echo():
        conn, _ = echo.accept()
        with conn:
            while data := conn.recv(65536):
                conn.sendall(data)

    threading.Thread(target=serve_echo, daemon=True).start()
    payload = os.urandom(5 * 1024 * 1024)
    s, status = _connect(proxy, target)
    with s:
        assert " 200 " in status, status
        threading.Thread(
            target=lambda: (s.sendall(payload), s.shutdown(socket.SHUT_WR)), daemon=True
        ).start()
        back = bytearray()
        while chunk := s.recv(65536):
            back += chunk
    echo.close()
    assert hashlib.sha256(back).hexdigest() == hashlib.sha256(payload).hexdigest()


# the worker


def test_the_plugin_gets_its_proxy_and_the_token_is_scrubbed(worker, monkeypatch):  # noqa: F811
    def leave(output):
        # what a careless plugin does: writes its environment into a file it leaves
        given = worker["popen"][-1][1]["env"]
        (output / "measures.json").write_text(ONE_MEASURE)
        (output / "notes.txt").write_text(given.get("HTTPS_PROXY", "") + "\n")

    worker["leave"] = leave
    _run()
    [(_argv, kwargs)] = worker["popen"]
    proxy_url = kwargs["env"].get("HTTPS_PROXY", "")
    assert proxy_url.startswith("http://run:") and proxy_url.endswith(
        "@127.0.0.1:3128"
    ), "the plugin is given no egress proxy"
    token = proxy_url[len("http://run:") : -len("@127.0.0.1:3128")]
    assert all(token.encode() not in body for body in worker["uploads"].values())


def test_a_root_worker_without_the_firewall_refuses_to_start_the_plugin(
    worker, monkeypatch
):  # noqa: F811
    e = egress()
    monkeypatch.setattr(e, "in_force", lambda run=None: False)
    monkeypatch.setattr(celery_tasks.os, "geteuid", lambda: 0)
    with contextlib.suppress(Exception):
        _run()
    assert worker["popen"] == []
    assert any(
        "network isolation is not in force" in str(f[-1]) for f in worker["failed"]
    ), worker["failed"]


def test_the_entrypoint_applies_the_firewall_and_starts_the_proxy():
    entrypoint = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"
    assert (
        entrypoint.exists()
    ), "the worker image has no entrypoint that sets the plugin firewall"
    text = entrypoint.read_text()
    assert "iptables" in text and "egress_proxy" in text


# what a run may reach, built from what the run names

HELPER_RULE = {
    "allowed_hosts": ["models.lan:11434"],
    "denied_addresses": ["172.20.0.5", "172.20.0.9"],
    "floor_hosts": [],
}


def test_a_runs_hosts_are_the_platform_its_helpers_and_the_operators(monkeypatch):
    e = egress()
    resolving(monkeypatch, {"platform": "172.20.0.5"})
    run = e.run_egress(
        "http://platform:8000",
        ["https://api.openai.com/v1", "http://models.lan:11434/v1"],
        [HELPER_RULE],
        ["raw.githubusercontent.com", ".hf.co"],
    )
    assert run.hosts == {
        "platform:8000",
        "api.openai.com:443",
        "models.lan:11434",
        "raw.githubusercontent.com",
        ".hf.co",
    }
    assert {"platform:8000", "models.lan:11434", "raw.githubusercontent.com"} <= set(
        run.rule["allowed_hosts"]
    )
    assert "platform:8000" in run.rule["floor_hosts"]
    # the platform is a service of the stack, yet the one a plugin must reach: its own address is not denied
    assert run.rule["denied_addresses"] == ["172.20.0.9"]


def test_a_run_without_helpers_reaches_the_platform_and_no_internal_host(monkeypatch):
    e = egress()
    resolving(monkeypatch, {"platform": "172.20.0.5", "redis": "172.20.0.9"})
    run = e.run_egress("http://platform:8000", [], [], [])
    assert e.decide("platform", 8000, run) == ipaddress.ip_address("172.20.0.5")
    with pytest.raises(e.Refused):
        e.decide("redis", 6379, run)


def test_an_operator_suffix_admits_its_domain_and_the_hosts_under_it_on_public_addresses_only(
    monkeypatch,
):
    e = egress()
    resolving(
        monkeypatch,
        {
            "cas-bridge.xethub.hf.co": "18.239.50.1",
            "evil.hf.co": "10.0.0.8",
            "nothf.co": "18.239.50.2",
            "hf.co": "18.239.50.3",
        },
    )
    run = e.run_egress("http://platform:8000", [], [], [".hf.co"])
    assert e.decide("hf.co", 443, run) == ipaddress.ip_address("18.239.50.3")
    assert e.decide("cas-bridge.xethub.hf.co", 443, run) == ipaddress.ip_address(
        "18.239.50.1"
    )
    with pytest.raises(e.Refused):
        e.decide("evil.hf.co", 443, run)
    with pytest.raises(e.Refused, match="not an address this test may reach"):
        e.decide("nothf.co", 443, run)


def test_the_operators_hosts_come_from_the_workers_setting(monkeypatch):
    e = egress()
    monkeypatch.setenv(
        "PLUGIN_EGRESS_ALLOWED_HOSTS", " Raw.GitHubUserContent.com, .hf.co ,,"
    )
    assert e.operator_hosts() == ["raw.githubusercontent.com", ".hf.co"]


def test_without_the_setting_plugins_may_reach_the_dataset_host(monkeypatch):
    e = egress()
    monkeypatch.delenv("PLUGIN_EGRESS_ALLOWED_HOSTS", raising=False)
    assert e.operator_hosts() == ["raw.githubusercontent.com"]


def test_an_operator_may_narrow_the_extra_hosts_to_none(monkeypatch):
    e = egress()
    monkeypatch.setenv("PLUGIN_EGRESS_ALLOWED_HOSTS", "")
    assert e.operator_hosts() == []


def test_setting_the_firewall_adds_only_the_rules_that_are_missing():
    e = egress()
    first = e.firewall_rules()[0]
    present = {(first[0], *first[2:])}
    ran = []

    class _Answer:
        def __init__(self, code):
            self.returncode = code

    def run(argv, **kw):
        ran.append(argv)
        rule = (argv[0], *argv[2:])
        if argv[1] == "-C":
            return _Answer(0 if rule in present else 1)
        present.add(rule)
        return _Answer(0)

    assert e.apply_firewall(run=run) is True
    added = [argv for argv in ran if argv[1] == "-A"]
    assert added == e.firewall_rules()[1:]
    ran.clear()
    assert e.apply_firewall(run=run) is True
    assert not [argv for argv in ran if argv[1] == "-A"]


def test_a_worker_without_iptables_has_no_firewall_in_force():
    e = egress()

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])

    assert e.in_force(run=missing) is False


def test_a_root_worker_whose_proxy_does_not_answer_refuses_to_start_the_plugin(
    worker, monkeypatch
):  # noqa: F811
    e = egress()
    monkeypatch.setattr(e, "in_force", lambda run=None: True)
    monkeypatch.setattr(e, "PROXY_PORT", _free_port())
    monkeypatch.setattr(celery_tasks.os, "geteuid", lambda: 0)
    with contextlib.suppress(Exception):
        _run()
    assert worker["popen"] == []
    assert any(
        "egress proxy is not running" in str(f[-1]) for f in worker["failed"]
    ), worker["failed"]


def test_the_runs_rule_file_is_gone_when_the_test_ends(worker):  # noqa: F811
    e = egress()
    _run()
    assert list(Path(e.POLICIES).glob("*.json")) == []
