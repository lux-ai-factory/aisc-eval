"""The run door (ADR 0009, I2): a plugin reaches a target through it, never the key.

A target may not be loopback (that is the platform), so these tests bind the mock
upstream on every interface and address it by the machine's own address, listing
that address in the door's allowlist. The mock records what it received, so the
tests assert on the call, not just the status.
"""
from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

from aisc_eval import door


def _local_address() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            address = sock.getsockname()[0]
    except OSError:
        return None
    return address if not address.startswith("127.") else None


class MockTarget(BaseHTTPRequestHandler):
    calls: list[dict] = []
    answer: dict = {"choices": [{"message": {"content": "hello from the target"}}]}

    def log_message(self, *args):
        pass

    def _reply(self, status: int, payload: dict, headers: dict | None = None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        if self.path.endswith("/collect"):
            self._reply(200, {"payload": {"text": f"collected {body.get('prompt')}"}})
        else:
            self._reply(200, type(self).answer)


@contextmanager
def _upstream():
    MockTarget.calls = []
    address = _local_address()
    server = HTTPServer(("0.0.0.0", 0), MockTarget)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = address or "127.0.0.1"
        yield f"http://{host}:{server.server_port}", f"{host}:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _door(spec: dict):
    server, runtime = door.build_server(spec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], runtime
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _slot(**overrides) -> dict:
    base = {
        "adapter": "openai", "version": "1", "face": "openai-chat",
        "protocol": "openai-chat", "config": {"model": "mock"},
        "token": "run-token", "allowed": [], "floor": [], "stack": [],
    }
    base.update(overrides)
    return base


def test_passthrough_forwards_with_the_real_key_and_scrubs_it_back():
    with _upstream() as (url, entry):
        spec = {
            "workspace": "",
            "slots": {"target": _slot(
                url=url, secret="sk-real", allowed=[entry],
            )},
        }
        with _door(spec) as (port, runtime):
            response = httpx.post(
                f"http://127.0.0.1:{port}/target/v1/chat/completions",
                headers={"Authorization": "Bearer run-token"},
                json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        assert MockTarget.calls[-1]["auth"] == "Bearer sk-real"
        assert MockTarget.calls[-1]["body"]["messages"][-1]["content"] == "hi"
        assert runtime.provenance()["target"]["calls"] == 1


def test_a_wrong_token_gets_401_and_never_calls_out():
    with _upstream() as (url, entry):
        spec = {"workspace": "", "slots": {"target": _slot(url=url, secret="sk-real", allowed=[entry])}}
        with _door(spec) as (port, _):
            response = httpx.post(
                f"http://127.0.0.1:{port}/target/v1/chat/completions",
                headers={"Authorization": "Bearer wrong"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 401
        assert MockTarget.calls == []


def test_an_unlisted_internal_host_is_refused_before_the_call():
    with _upstream() as (url, _entry):
        spec = {"workspace": "", "slots": {"target": _slot(url=url, secret="sk-real")}}
        with _door(spec) as (port, _):
            response = httpx.post(
                f"http://127.0.0.1:{port}/target/v1/chat/completions",
                headers={"Authorization": "Bearer run-token"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 403
        assert response.json()["error"] == "internal_not_allowed"
        assert MockTarget.calls == []


def test_the_target_key_never_appears_in_the_response():
    with _upstream() as (url, entry):
        MockTarget.answer = {"choices": [{"message": {"content": "the key is sk-real"}}]}
        try:
            spec = {"workspace": "", "slots": {"target": _slot(url=url, secret="sk-real", allowed=[entry])}}
            with _door(spec) as (port, _):
                response = httpx.post(
                    f"http://127.0.0.1:{port}/target/v1/chat/completions",
                    headers={"Authorization": "Bearer run-token"},
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
        finally:
            MockTarget.answer = {"choices": [{"message": {"content": "hello from the target"}}]}
        assert "sk-real" not in response.text
        assert "***" in response.text


def test_a_rest_adapter_maps_a_nested_response_path():
    with _upstream() as (url, entry):
        spec = {
            "workspace": "",
            "slots": {"bespoke": _slot(
                adapter="rest", face="openai-chat", protocol="rest", url=url,
                config={
                    "path": "/collect", "method": "POST",
                    "body_template": {"prompt": "{{input}}"}, "response_path": "payload.text",
                },
                secret="", allowed=[entry],
            )},
        }
        with _door(spec) as (port, _):
            response = httpx.post(
                f"http://127.0.0.1:{port}/bespoke/v1/chat/completions",
                headers={"Authorization": "Bearer run-token"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "collected hello"
        assert MockTarget.calls[-1]["path"] == "/collect"


def test_provenance_is_written_to_the_workspace(tmp_path):
    with _upstream() as (url, entry):
        spec = {
            "workspace": str(tmp_path),
            "slots": {"target": _slot(
                url=url, secret="sk-real", allowed=[entry], config_hash="abc",
            )},
        }
        with _door(spec) as (port, runtime):
            httpx.post(
                f"http://127.0.0.1:{port}/target/v1/chat/completions",
                headers={"Authorization": "Bearer run-token"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            runtime.write_provenance()
        written = json.loads((tmp_path / "door_provenance.json").read_text())
        assert written["target"]["adapter"] == "openai"
        assert written["target"]["config_hash"] == "abc"
        assert written["target"]["calls"] == 1
