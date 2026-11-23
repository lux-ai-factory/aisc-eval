#!/usr/bin/env python3
"""The run door (ADR 0009, I2): one local process, one route per endpoint slot.

A plugin never receives the assessed system's key. It is handed a local URL and a
per-run token instead; every call goes to this door, which holds the real key,
checks the allowed-hosts rule immediately before the call out, and scrubs the key
out of everything it returns. The door runs **in the plugin's own venv**, so an
adapter a plugin ships imports; it is a separate process, started before the
plugin and stopped with it, so nothing it holds outlives the run.

It is handed its spec on stdin as JSON::

    {
      "workspace": "/run/workspace",
      "slots": {
        "<slot>": {
          "adapter": "openai", "version": "1",
          "module": "", "class": "",          # empty => a builtin adapter by id
          "face": "openai-chat",              # the protocol the plugin speaks
          "protocol": "openai-chat",          # the target's protocol
          "url": "https://api.example/v1",
          "config": {"model": "gpt-4o"},
          "secret": "sk-real", "secret_headers": {"X-Api-Key": "..."},
          "token": "run-token",
          "allowed": ["10.0.0.5:11434"], "floor": [], "stack": ["postgres"],
          "config_hash": "…", "timeout_s": 60, "max_concurrency": 8
        }
      }
    }

It prints ``{"port": <n>}`` on stdout once it is listening, and writes its
provenance (per slot: adapter, version, url, config_hash, calls, errors) to
``<workspace>/door_provenance.json`` as it goes.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import Any
from urllib.parse import urlsplit

import httpx

from aisc_plugin_interface.adapters import BUILTIN_ADAPTERS, EndpointAdapter, Secret, Turn
from aisc_plugin_interface.adapters import faces
from aisc_plugin_interface.adapters import hosts as host_rules

MASK = "***"

#: Headers the door must not forward verbatim (hop-by-hop, or the door's own auth).
_HOP_HEADERS = frozenset(
    {"host", "content-length", "connection", "authorization", "transfer-encoding", "accept-encoding"}
)


def _adapter_class(spec: dict) -> type[EndpointAdapter]:
    module = spec.get("module") or ""
    if module:
        cls = getattr(import_module(module), spec["class"])
        if not issubclass(cls, EndpointAdapter):
            raise TypeError(f"{spec['class']} is not an endpoint adapter")
        return cls
    for cls in BUILTIN_ADAPTERS:
        if getattr(cls, "id", "") == spec.get("adapter"):
            return cls
    raise KeyError(f"builtin adapter {spec.get('adapter')!r} not found")


@dataclass
class _Slot:
    name: str
    adapter: EndpointAdapter
    adapter_id: str
    version: str
    face: str
    protocol: str
    url: str
    secret: Secret
    token: str
    allowed: set[str]
    floor: set[str]
    stack: set[str]
    config_hash: str = ""
    timeout_s: float = 60.0
    max_concurrency: int = 8
    calls: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def target_of(self):
        return _Target(self.url, self.protocol)


@dataclass(frozen=True)
class _Target:
    url: str = ""
    protocol: str = ""


def _secret_values(slot: _Slot) -> list[str]:
    values = [slot.secret.key or "", *slot.secret.headers.values()]
    return [v for v in values if v]


def _scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, MASK)
    return text


class DoorRuntime:
    def __init__(self, spec: dict):
        self.workspace = spec.get("workspace", "")
        self.slots: dict[str, _Slot] = {}
        for name, slot_spec in (spec.get("slots") or {}).items():
            cls = _adapter_class(slot_spec)
            self.slots[name] = _Slot(
                name=name,
                adapter=cls(slot_spec.get("config") or {}),
                adapter_id=slot_spec.get("adapter", ""),
                version=str(slot_spec.get("version", "")),
                face=slot_spec.get("face", ""),
                protocol=slot_spec.get("protocol", ""),
                url=slot_spec.get("url", ""),
                secret=Secret(
                    key=slot_spec.get("secret") or None,
                    headers=dict(slot_spec.get("secret_headers") or {}),
                ),
                token=slot_spec.get("token", ""),
                allowed=set(slot_spec.get("allowed") or []),
                floor=set(slot_spec.get("floor") or []),
                stack=set(slot_spec.get("stack") or []),
                config_hash=slot_spec.get("config_hash", ""),
                timeout_s=float(slot_spec.get("timeout_s") or 60),
                max_concurrency=int(slot_spec.get("max_concurrency") or 8),
            )
        self._semaphores = {
            name: threading.Semaphore(slot.max_concurrency) for name, slot in self.slots.items()
        }
        self._provenance_lock = threading.Lock()

    # ---- provenance -------------------------------------------------- #
    def provenance(self) -> dict:
        return {
            name: {
                "adapter": slot.adapter_id,
                "version": slot.version,
                "url": slot.url,
                "config_hash": slot.config_hash,
                "calls": slot.calls,
                "errors": slot.errors,
            }
            for name, slot in self.slots.items()
        }

    def write_provenance(self) -> None:
        if not self.workspace:
            return
        try:
            path = os.path.join(self.workspace, "door_provenance.json")
            with self._provenance_lock, open(path, "w") as handle:
                json.dump(self.provenance(), handle)
        except OSError:
            pass

    def record(self, slot: _Slot, *, error: bool) -> None:
        with self._provenance_lock:
            slot.calls += 1
            if error:
                slot.errors += 1
        self.write_provenance()

    # ---- calls ------------------------------------------------------- #
    def refusal(self, slot: _Slot, url: str) -> tuple[str, str] | None:
        return host_rules.refusal_for(
            url, allowed=slot.allowed, floor=slot.floor, stack=slot.stack
        )

    def call(self, slot: _Slot, *, path: str, method: str, headers: dict, body: bytes):
        """Return (status, headers, body_iter) for one plugin call."""
        if slot.adapter.passthrough:
            json_body, text_body = _decode_body(headers, body)
            request = slot.adapter.rewrite(
                slot.target_of(), slot.secret, path=path, method=method,
                headers=headers, json_body=json_body, text_body=text_body,
            )
        else:
            face = faces.FACES.get(slot.face)
            if face is None:
                raise KeyError(f"no face for {slot.face!r}")
            parse_request, _ = face
            document = json.loads(body or b"{}")
            turn = parse_request(document)
            request = slot.adapter.build_request(slot.target_of(), slot.secret, turn=turn)

        refusal = self.refusal(slot, request.url)
        if refusal is not None:
            code, detail = refusal
            return 403, {"Content-Type": "application/json"}, json.dumps(
                {"error": code, "detail": detail}
            ).encode()

        return self._send(slot, request)

    def _send(self, slot: _Slot, request) -> tuple[int, dict, Any]:
        secrets = _secret_values(slot)
        content = None
        if request.json_body is not None:
            content = json.dumps(request.json_body).encode()
        elif request.text_body is not None:
            content = request.text_body.encode()
        try:
            with self._semaphores[slot.name]:
                with httpx.Client(timeout=slot.timeout_s) as client:
                    response = client.request(
                        request.method, request.url, headers=request.headers, content=content
                    )
        except httpx.HTTPError as exc:
            self.record(slot, error=True)
            return 502, {"Content-Type": "application/json"}, json.dumps(
                {"error": "unreachable", "detail": str(exc)}
            ).encode()

        if slot.adapter.passthrough:
            self.record(slot, error=response.status_code >= 500)
            headers = _scrubbed_headers(response.headers, secrets)
            return response.status_code, headers, _scrub(response.text, secrets)

        self.record(slot, error=response.status_code >= 500)
        answer = slot.adapter.parse_response(response.status_code, response.text, slot.target_of())
        if answer.ok and not answer.refused:
            answer = _scrub_answer(answer, secrets)
        _, render_response = faces.FACES[slot.face]
        rendered = render_response(answer, model=slot.adapter.config.model)
        body = _scrub(json.dumps(rendered), secrets)
        return 200, {"Content-Type": "application/json"}, body


def _scrub_answer(answer, secrets: list[str]):
    if isinstance(answer.answer, str):
        from dataclasses import replace

        return replace(answer, answer=_scrub(answer.answer, secrets))
    return answer


def _scrubbed_headers(headers, secrets: list[str]) -> dict[str, str]:
    out = {}
    for name, value in headers.items():
        if name.lower() in _HOP_HEADERS:
            continue
        out[name] = _scrub(value, secrets)
    return out


def _decode_body(headers: dict, body: bytes):
    content_type = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
    if "json" in content_type:
        try:
            return json.loads(body or b"{}"), None
        except ValueError:
            return None, body.decode("utf-8", "replace")
    return None, body.decode("utf-8", "replace") if body else None


class _Handler(BaseHTTPRequestHandler):
    runtime: DoorRuntime = None  # type: ignore[assignment]

    def log_message(self, *args):  # the worker captures stdout; keep it clean
        pass

    def _reply(self, status: int, headers: dict, body: Any) -> None:
        payload = body if isinstance(body, bytes) else str(body).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle(self, method: str) -> None:
        parsed = urlsplit(self.path)
        parts = parsed.path.lstrip("/").split("/", 1)
        slot_name = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        slot = self.runtime.slots.get(slot_name)
        if slot is None:
            self._reply(404, {"Content-Type": "application/json"}, b'{"error":"unknown_slot"}')
            return
        token = self.headers.get("Authorization", "")
        if token != f"Bearer {slot.token}":
            self._reply(401, {"Content-Type": "application/json"}, b'{"error":"unauthorized"}')
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_HEADERS
        }
        try:
            status, out_headers, out_body = self.runtime.call(
                slot, path=rest, method=method, headers=headers, body=body
            )
        except Exception as exc:  # a bad adapter/config is the run's failure, not the door's
            self.runtime.record(slot, error=True)
            self._reply(
                500, {"Content-Type": "application/json"},
                json.dumps({"error": "door_error", "detail": str(exc)}),
            )
            return
        self._reply(status, out_headers, out_body)

    def do_GET(self):  # noqa: N802
        self._handle("GET")

    def do_POST(self):  # noqa: N802
        self._handle("POST")

    def do_PUT(self):  # noqa: N802
        self._handle("PUT")

    def do_PATCH(self):  # noqa: N802
        self._handle("PATCH")

    def do_DELETE(self):  # noqa: N802
        self._handle("DELETE")


def build_server(spec: dict) -> tuple[ThreadingHTTPServer, DoorRuntime]:
    """A bound door server and its runtime (the tests drive this directly)."""
    runtime = DoorRuntime(spec)
    handler = type("_BoundHandler", (_Handler,), {"runtime": runtime})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    return server, runtime


def serve(spec: dict) -> int:
    server, runtime = build_server(spec)
    port = server.server_address[1]

    def _stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(json.dumps({"port": port}), flush=True)
    runtime.write_provenance()
    try:
        server.serve_forever()
    finally:
        runtime.write_provenance()
    return port


def main() -> None:
    spec = json.loads(sys.stdin.read() or "{}")
    serve(spec)


if __name__ == "__main__":
    main()
