"""The egress proxy: the only way out for a plugin test (see egress).

A forward proxy on the worker's loopback. Each request names its test with `Proxy-Authorization: Basic
run:<token>` (407 without a token of a running test). CONNECT host:port opens a tunnel and a plain request in
absolute form (GET http://host/path) is passed on; either way the host is checked by the test's rule
(egress.decide, 403 with the reason when refused) and the proxy connects to the address it checked, never to a
name resolved again. Nothing is added to what passes: no Forwarded or X-Forwarded-For header, and the
Proxy-Authorization header is not passed on. A connection idle for 10 minutes is closed.

`python -m aisc_eval.egress_proxy` serves on 127.0.0.1:3128; started as root, it gives up root for the
aisc-egress user before it serves (the image's entrypoint starts it).
"""

import asyncio
import base64
import binascii
import hashlib
import os
import pwd
import sys
import time
import urllib.parse
from pathlib import Path

from aisc_eval import egress

#: seconds a connection may stay idle, both ways, before it is closed (an LLM may think for minutes)
IDLE_TIMEOUT_S = 600
#: seconds a client has to send its request line and headers
HEAD_TIMEOUT_S = 30
#: seconds to open the connection to the host
CONNECT_TIMEOUT_S = 30
#: the largest request or response head read
HEAD_LIMIT = 64 * 1024
CHUNK = 64 * 1024
#: headers between a client and the proxy, or about one connection only, never passed on
_HOP_HEADERS = {
    "proxy-authorization",
    "proxy-connection",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
}


def _log(text: str) -> None:
    print(f"egress proxy: {text}", file=sys.stderr, flush=True)


def _answer(status: str, text: str = "", extra: str = "") -> bytes:
    body = (text + "\n").encode() if text else b""
    return (
        f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(body)}\r\n"
        f"{extra}Connection: close\r\n\r\n"
    ).encode() + body


def _token(headers: list[tuple[str, str]]) -> str | None:
    for name, value in headers:
        if name.lower() == "proxy-authorization":
            scheme, _sep, encoded = value.strip().partition(" ")
            if scheme.lower() != "basic":
                return None
            try:
                user, _sep, token = (
                    base64.b64decode(encoded.strip(), validate=True)
                    .decode()
                    .partition(":")
                )
            except (binascii.Error, UnicodeDecodeError):
                return None
            return token if user == "run" and token else None
    return None


def _authority(text: str, default_port: int) -> tuple[str, int]:
    parts = urllib.parse.urlsplit("//" + text)
    if not parts.hostname:
        raise ValueError(text)
    return parts.hostname.lower(), parts.port or default_port


class _Idle:
    """When either way of a connection last carried bytes."""

    def __init__(self):
        self.last = time.monotonic()


async def _copy(
    src: asyncio.StreamReader, dst: asyncio.StreamWriter, idle: _Idle
) -> None:
    """Bytes from src to dst until src ends, then end dst's sending side; stops when the connection was idle
    for IDLE_TIMEOUT_S both ways."""
    try:
        while True:
            try:
                data = await asyncio.wait_for(src.read(CHUNK), IDLE_TIMEOUT_S)
            except TimeoutError:
                if time.monotonic() - idle.last >= IDLE_TIMEOUT_S:
                    return
                continue
            if not data:
                break
            idle.last = time.monotonic()
            dst.write(data)
            await dst.drain()
        if dst.can_write_eof():
            dst.write_eof()
    except (ConnectionError, OSError):
        return


async def _pipe(a_reader, a_writer, b_reader, b_writer) -> None:
    idle = _Idle()
    await asyncio.gather(
        _copy(a_reader, b_writer, idle), _copy(b_reader, a_writer, idle)
    )


def _head(raw: bytes) -> tuple[str, list[tuple[str, str]]]:
    lines = raw.decode("latin-1").split("\r\n")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, _sep, value = line.partition(":")
        headers.append((name.strip(), value.strip()))
    return lines[0], headers


async def _forward_response_head(
    upstream: asyncio.StreamReader, client: asyncio.StreamWriter
) -> None:
    """The host's status line and headers, with its connection headers replaced by Connection: close: one
    request per connection, so a client never sends a second one for another host down this one."""
    raw = await asyncio.wait_for(upstream.readuntil(b"\r\n\r\n"), IDLE_TIMEOUT_S)
    status, headers = _head(raw)
    kept = [f"{n}: {v}" for n, v in headers if n.lower() not in _HOP_HEADERS]
    client.write(
        ("\r\n".join([status, *kept, "Connection: close"]) + "\r\n\r\n").encode(
            "latin-1"
        )
    )
    await client.drain()


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, policies: Path
) -> None:
    upstream_writer = None
    try:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
        except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            TimeoutError,
            ConnectionError,
        ):
            return
        request_line, headers = _head(raw)
        try:
            method, target, version = request_line.split(" ", 2)
        except ValueError:
            writer.write(_answer("400 Bad Request", "not an HTTP request"))
            return
        token = _token(headers)
        run = egress.load(token, policies) if token else None
        if run is None:
            writer.write(
                _answer(
                    "407 Proxy Authentication Required",
                    "this proxy serves running plugin tests only",
                    'Proxy-Authenticate: Basic realm="aisc-egress"\r\n',
                )
            )
            return
        test = hashlib.sha256(token.encode()).hexdigest()[:8]
        try:
            if method.upper() == "CONNECT":
                host, port = _authority(target, 443)
                path = None
            else:
                parts = urllib.parse.urlsplit(target)
                if parts.scheme != "http" or not parts.hostname:
                    writer.write(
                        _answer(
                            "400 Bad Request",
                            "a request through this proxy names an http:// URL",
                        )
                    )
                    return
                host, port = parts.hostname.lower(), parts.port or 80
                path = urllib.parse.urlunsplit(
                    ("", "", parts.path or "/", parts.query, "")
                )
        except ValueError:
            writer.write(_answer("400 Bad Request", "the request does not name a host"))
            return
        loop = asyncio.get_running_loop()
        try:
            address = await loop.run_in_executor(None, egress.decide, host, port, run)
        except egress.Refused as refused:
            _log(f"test {test}: refused {host}:{port}: {refused}")
            writer.write(_answer("403 Forbidden", str(refused)))
            return
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(str(address), port, limit=HEAD_LIMIT),
                CONNECT_TIMEOUT_S,
            )
        except (OSError, TimeoutError) as exc:
            writer.write(
                _answer(
                    "502 Bad Gateway",
                    f"{host}:{port} could not be reached ({type(exc).__name__})",
                )
            )
            return
        if path is None:
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
        else:
            kept = [f"{n}: {v}" for n, v in headers if n.lower() not in _HOP_HEADERS]
            upstream_writer.write(
                (
                    "\r\n".join(
                        [f"{method} {path} {version}", *kept, "Connection: close"]
                    )
                    + "\r\n\r\n"
                ).encode("latin-1")
            )
            await upstream_writer.drain()
            # the request's body, if any, follows in the pipe; the answer's head is rewritten first
            body = asyncio.ensure_future(_copy(reader, upstream_writer, _Idle()))
            try:
                await _forward_response_head(upstream_reader, writer)
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                TimeoutError,
                ConnectionError,
            ):
                body.cancel()
                writer.write(
                    _answer("502 Bad Gateway", f"{host}:{port} gave no HTTP answer")
                )
                return
            await _copy(upstream_reader, writer, _Idle())
            body.cancel()
            return
        await _pipe(reader, writer, upstream_reader, upstream_writer)
    except Exception as exc:  # one connection's failure never stops the proxy
        _log(f"a connection failed: {type(exc).__name__}")
    finally:
        for stream in (writer, upstream_writer):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


async def _serve(host: str, port: int, policies: Path) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, policies),
        host,
        port,
        limit=HEAD_LIMIT,
        reuse_address=True,
    )
    async with server:
        await server.serve_forever()


def serve(
    host: str = egress.PROXY_HOST, port: int | None = None, policies: Path | None = None
) -> None:
    """Serve until the process ends."""
    asyncio.run(
        _serve(host, port or egress.PROXY_PORT, Path(policies or egress.POLICIES))
    )


def _become(name: str) -> None:
    """Give up root for `name`, its group only."""
    entry = pwd.getpwnam(name)
    os.setgroups([])
    os.setgid(entry.pw_gid)
    os.setuid(entry.pw_uid)
    if os.getuid() == 0 or os.geteuid() == 0:
        raise RuntimeError("the egress proxy is still root")


if __name__ == "__main__":
    if os.geteuid() == 0:
        _become(egress.EGRESS_USER)
    _log(f"serving on {egress.PROXY_HOST}:{egress.PROXY_PORT}")
    serve()
