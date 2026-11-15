"""What a plugin test may reach on the network, and the firewall that leaves it no other way out.

A plugin test runs as a leased user (plugin_process.lease_run_user). The firewall lets the pool's uids open
one connection only, to the egress proxy on the worker's loopback; everything else from them, IPv6 and DNS
included, is rejected. The proxy (egress_proxy) asks each test for its token and admits only the hosts of
that test: the platform (its facade and resolve routes), the endpoints of its helper and LLM inputs, and the
hosts the operator lists in PLUGIN_EGRESS_ALLOWED_HOSTS. Each is checked by the network rule of the plugin
interface (connections.checked_addresses) on the address the proxy then connects to, so a name that resolves
again to something else is caught at the connection: the project's allowed internal hosts, the stack's
services and metadata never, loopback only for the deployment's own hosts, no other non-global range.

The worker writes each test's hosts and rule to a file named by the hash of its token (register), readable
by the proxy only, and removes it when the test ends (unregister).

`python -m aisc_eval.egress` sets the firewall (the image's entrypoint runs it as root, once).
"""

import grp
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from aisc_plugin_interface.connections import BlockedAddress, checked_addresses

from aisc_eval.plugin_process import RUN_UIDS

#: where the worker leaves each running test's rule for the proxy: root's, readable by the proxy's group
POLICIES = Path("/run/aisc-egress")
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 3128
#: the user the proxy runs as: neither root nor a run user, so the firewall leaves it alone
EGRESS_USER = "aisc-egress"
#: the operator's extra hosts every plugin test may reach (a dataset or model download): host, host:port,
#: or .domain for the domain and every host under it (on a public address only)
OPERATOR_HOSTS_SETTING = "PLUGIN_EGRESS_ALLOWED_HOSTS"
#: the extra hosts when the operator sets none: where StrongREJECT reads its datasets (a read-only host)
DEFAULT_OPERATOR_HOSTS = "raw.githubusercontent.com"


class Refused(Exception):
    """The test may not reach this host, or not on the address it resolves to; the text says why."""


@dataclass(frozen=True)
class RunEgress:
    #: host, host:port or .domain entries the test may reach
    hosts: frozenset[str]
    #: the network rule they are checked by: allowed_hosts, denied_addresses, floor_hosts (connections)
    rule: dict


def _host_port(url: str) -> str | None:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host or parts.scheme not in ("http", "https"):
        return None
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
    return f"{host}:{port}"


def _addresses_of(host_port: str) -> set:
    host, _sep, port = host_port.rpartition(":")
    try:
        infos = socket.getaddrinfo(host, int(port), proto=socket.IPPROTO_TCP)
    except (OSError, ValueError):
        return set()
    return {ipaddress.ip_address(info[4][0]) for info in infos}


def run_egress(
    platform_url: str | None,
    endpoints: list[str],
    helper_rules: list[dict],
    operator_hosts: list[str],
) -> RunEgress:
    """What one test may reach: the platform, the endpoints its helper and LLM inputs name, and the operator's
    hosts, under the network rule its helpers came with from the platform (the project's allowed internal hosts,
    the stack's addresses, the floor). The platform and the operator's hosts are allowed and of the floor. The
    platform's own address is a stack service's, and so on the rule's denied list: it is taken off it here."""
    platform = _host_port(platform_url) if platform_url else None
    hosts = {h for h in (_host_port(url) for url in endpoints) if h}
    hosts.update(h.lower() for h in operator_hosts)
    own = [h.lower() for h in operator_hosts if not h.startswith(".")]
    if platform:
        hosts.add(platform)
        own.insert(0, platform)
    allowed: list[str] = []
    denied: list[str] = []
    floor: list[str] = []
    for rule in helper_rules:
        allowed += [h for h in rule.get("allowed_hosts") or [] if h not in allowed]
        denied += [a for a in rule.get("denied_addresses") or [] if a not in denied]
        floor += [h for h in rule.get("floor_hosts") or [] if h not in floor]
    allowed += [h for h in own if h not in allowed]
    floor += [h for h in own if h not in floor]
    if platform:
        mine = _addresses_of(platform)
        denied = [a for a in denied if ipaddress.ip_address(a) not in mine]
    return RunEgress(
        frozenset(hosts),
        {"allowed_hosts": allowed, "denied_addresses": denied, "floor_hosts": floor},
    )


def operator_hosts() -> list[str]:
    """PLUGIN_EGRESS_ALLOWED_HOSTS, comma separated, lower case; DEFAULT_OPERATOR_HOSTS when it is not set
    (set empty, it narrows the list to none)."""
    raw = os.environ.get(OPERATOR_HOSTS_SETTING, DEFAULT_OPERATOR_HOSTS)
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _may_reach(host: str, port: int, hosts: frozenset[str]) -> bool:
    if host in hosts or f"{host}:{port}" in hosts:
        return True
    return any(
        entry.startswith(".") and (host.endswith(entry) or host == entry[1:])
        for entry in hosts
    )


def decide(host: str, port: int, egress: RunEgress) -> ipaddress._BaseAddress:
    """The address the proxy connects to for host:port, checked now by the run's rule; Refused with the
    reason when the test may not reach it."""
    host = host.lower().strip("[]")
    if not _may_reach(host, port, egress.hosts):
        raise Refused(
            f"{host} is not an address this test may reach (an operator can add it to {OPERATOR_HOSTS_SETTING})"
        )
    rule = egress.rule
    try:
        addresses = checked_addresses(
            host,
            port,
            rule.get("allowed_hosts") or [],
            rule.get("denied_addresses") or [],
            rule.get("floor_hosts") if rule.get("floor_hosts") is not None else [],
        )
    except BlockedAddress as exc:
        raise Refused(str(exc)) from None
    return addresses[0]


def _policy_file(token: str, policies: Path) -> Path:
    return policies / (hashlib.sha256(token.encode()).hexdigest() + ".json")


def register(token: str, egress: RunEgress, policies: Path | None = None) -> Path:
    """Leave the test's hosts and rule for the proxy, under the hash of its token (never the token): root's
    and readable by the proxy's group only."""
    folder = Path(policies or POLICIES)
    path = _policy_file(token, folder)
    staged = path.with_suffix(".tmp")
    body = json.dumps({"hosts": sorted(egress.hosts), "rule": egress.rule}).encode()
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(body)
    if os.geteuid() == 0:
        os.chown(staged, 0, grp.getgrnam(EGRESS_USER).gr_gid)
    os.chmod(staged, 0o640)
    os.replace(staged, path)
    return path


def unregister(token: str, policies: Path | None = None) -> None:
    _policy_file(token, Path(policies or POLICIES)).unlink(missing_ok=True)


def load(token: str, policies: Path | None = None) -> RunEgress | None:
    """The rule a running test left under its token, or None (no such test)."""
    try:
        data = json.loads(_policy_file(token, Path(policies or POLICIES)).read_text())
    except (OSError, ValueError):
        return None
    return RunEgress(frozenset(data.get("hosts") or ()), data.get("rule") or {})


def proxy_env(token: str) -> dict[str, str]:
    """The variables that send a plugin's HTTP libraries (requests, httpx, urllib, aiohttp with trust_env,
    Node's fetch) through the proxy, with its test's token."""
    url = f"http://run:{token}@{PROXY_HOST}:{PROXY_PORT}"
    return {
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "http_proxy": url,
        "https_proxy": url,
        "NO_PROXY": "",
        "no_proxy": "",
        "NODE_USE_ENV_PROXY": "1",
    }


def firewall_rules(pool: range = RUN_UIDS, port: int = PROXY_PORT) -> list[list[str]]:
    """The run users may open a TCP connection to the proxy and nothing else, on IPv4 and IPv6 alike."""
    owners = f"{pool[0]}-{pool[-1]}"
    return [
        ["iptables", "-A", "OUTPUT", "-m", "owner", "--uid-owner", owners, "-d", PROXY_HOST,
         "-p", "tcp", "--dport", str(port), "-j", "ACCEPT"],
        ["iptables", "-A", "OUTPUT", "-m", "owner", "--uid-owner", owners, "-j", "REJECT"],
        ["ip6tables", "-A", "OUTPUT", "-m", "owner", "--uid-owner", owners, "-j", "REJECT"],
    ]  # fmt: skip


def _present(rule: list[str], run) -> bool:
    try:
        return run([rule[0], "-C", *rule[2:]], capture_output=True).returncode == 0
    except OSError:
        return False


def in_force(run=subprocess.run) -> bool:
    """Whether every rule of the firewall is set (iptables -C): false without iptables or NET_ADMIN."""
    return all(_present(rule, run) for rule in firewall_rules())


def apply_firewall(run=subprocess.run) -> bool:
    """Add the rules that are missing, in order, then say whether all are in force."""
    for rule in firewall_rules():
        if not _present(rule, run):
            try:
                run(rule, capture_output=True)
            except OSError:
                return False
    return in_force(run)


def proxy_answers(timeout: float = 2.0) -> bool:
    """Whether the proxy takes connections."""
    try:
        with socket.create_connection((PROXY_HOST, PROXY_PORT), timeout):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(0 if apply_firewall() else 1)
