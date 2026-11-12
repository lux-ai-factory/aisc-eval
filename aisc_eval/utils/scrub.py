"""Remove secret values from anything that leaves the worker (I7).

The worker decrypts run secrets to hand them to a plugin. The same values can
come back out in plugin stdout, in an error, or in a log line. Every outbound
path that can carry them runs through here first.
"""

from __future__ import annotations

from typing import Iterable

MASK = "***"


def scrub_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace every occurrence of a secret value in ``text`` with a mask."""
    if not text:
        return text
    # Longest first: a secret that contains another secret must be masked whole.
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return text


def scrub_bytes(content: bytes, secrets: Iterable[str]) -> bytes:
    """Scrub a byte payload when it is text; return binary payloads untouched."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return scrub_secrets(text, secrets).encode("utf-8")
