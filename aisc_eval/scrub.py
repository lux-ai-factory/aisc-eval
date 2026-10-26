"""Failure text cleared of secrets before the worker reports, raises or uploads it.

`scrub_text` replaces the values the worker knows it handed the plugin (`known_secrets`: the project's secrets,
the run's pass, the worker's own model keys, the secret-named values of the plugin configuration), then anything
shaped like a known kind of key, then the key-like value of a `password=...` style assignment.

Stdlib only. The engine's backend (aisc_backend/config_secrets.py) carries the same text rule, and a root test
keeps the two equal.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

#: What a secret in text becomes.
REDACTED = "[redacted]"
#: A known secret shorter than this is never searched for (it would match ordinary words).
MIN_KNOWN_LENGTH = 8

#: A key holding one of these words is a secret in a ledger copy (`Authorization` and `Cookie` as header fields).
LEDGER_SECRET_NAME = re.compile(
    r"key|token|secret|password|credential|authorization|cookie", re.IGNORECASE
)
#: A value that looks like a credential after `Bearer`, `Basic`, `Token` or a bare `token=`: 16 characters or more
#: of the base64, hex and URL-safe alphabets, no space (`the bearer of bad news`, `token=EOF` are not secrets).
CREDENTIAL = r"[A-Za-z0-9+/=_.~-]{16,}"
#: Strings shaped like a known kind of key, wherever they are: provider key prefixes, JWTs, a credential-shaped
#: `Bearer` value, a credential-shaped `Basic` or `Token` value, bare or in an `Authorization:` header (group 1
#: keeps the header name), the `user:password` of a URL and a whole PEM private-key block. Not caught: a
#: base64-encoded key (other than a Basic value), a key split across lines, a key with no known prefix outside an
#: assignment.
SECRET_SHAPE = re.compile(
    r"(?i)\bbearer\s+" + CREDENTIAL + r"|\bsk-[A-Za-z0-9_-]{16,}"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*"
    r"|\bhf_[A-Za-z0-9]{20,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bxox[abpr]-[A-Za-z0-9-]{10,}"
    r"|\bAIza[0-9A-Za-z_-]{35}"
    r"|(\bauthorization\s*:\s*)?\b(?:basic|token)\s+"
    + CREDENTIAL
    + r"|(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"
    r"|-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----"
)
#: `password=...`, `api_key: "..."`, `api_key%3D...` and the like in text. Group 1 is the name, which may carry
#: any prefix ending in `_` or at a lower-to-upper case step (`judge_api_key`, `DB_PASSWORD`, `judgeApiKey`), but
#: nothing after it (`api_key_env=`, `password_min_length=` are not secrets); group 3 is the value, up to the next
#: whitespace or the closing quote. A bare `token` or `pwd` counts only before `=` (`unexpected token: }` and
#: `pwd: /home` are not secrets). The value is redacted only when it looks like a key (see `_key_like`), so
#: `password: it must contain a digit` stays. Not caught: a value that reads like a word (no digit, and under 16
#: characters or all in one case, such as `password=letmein`), and a quoted value of several words (a multi-word
#: passphrase such as `password = 'correct horse battery staple'`), which reads like prose.
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b((?:[a-z0-9-]*_|[a-z0-9]*(?-i:[a-z0-9])(?=(?-i:[A-Z])))?"
    r"(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|refresh[_-]?token|secret[_-]?key"
    r"|(?:aws[_-]?)?secret[_-]?access[_-]?key|client[_-]?secret|password|passwd"
    r"|(?:token|pwd)(?=[\"']?\s*(?:=|%3d))))"
    r"([\"']?\s*(?:[=:]|%3d)\s*[\"']?)"
    r"((?!\[redacted\])(?:(?<=\")[^\"\s]+(?=\")|(?<=')[^'\s]+(?=')|[^\s\"',;]+))"
)

#: The worker's own keys a plugin is handed.
WORKER_KEY_NAMES = ("AISC_TARGET_API_KEY", "API_KEY_OPENAI", "OPENAI_API_KEY")


def _named_values(value, out: list[str]) -> None:
    """Every string under a secret-named key of `value`, at any depth."""
    if isinstance(value, dict):
        for name, inner in value.items():
            if LEDGER_SECRET_NAME.search(str(name)):
                _strings(inner, out)
            else:
                _named_values(inner, out)
    elif isinstance(value, list):
        for item in value:
            _named_values(item, out)


def _strings(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for inner in value.values():
            _strings(inner, out)
    elif isinstance(value, list):
        for item in value:
            _strings(item, out)


def known_secrets(
    secret_env: Mapping[str, str],
    worker_env: Mapping[str, str],
    plugin_config,
    run_pass: str | None,
) -> list[str]:
    """The secret values a run hands its plugin, 8 characters or more, longest first."""
    from aisc_eval.celery_tasks import (
        OPENAI_PLACEHOLDER,
    )  # the worker's module; imported late, it imports this

    values = list(secret_env.values())
    if run_pass:
        values.append(run_pass)
    values.extend(
        worker_env[name]
        for name in WORKER_KEY_NAMES
        if worker_env.get(name) and worker_env[name] != OPENAI_PLACEHOLDER
    )
    _named_values(plugin_config, values)
    return _ordered_known(values)


def _ordered_known(known: Iterable[str]) -> list[str]:
    values = {
        value
        for value in known
        if isinstance(value, str) and len(value) >= MIN_KNOWN_LENGTH
    }
    return sorted(values, key=lambda value: (-len(value), value))


_CREDENTIAL_VALUE = re.compile(CREDENTIAL)
_STRONG_TOKEN_NAME = re.compile(r"(?i)(?:access|auth|refresh)[_-]?token$")


def _key_like(name: str, value: str) -> bool:
    """Whether an assigned value looks like a key rather than a word: after a bare `token` it must look like a
    credential (16 characters or more); after any other name it holds a digit, or is 16 characters or more with
    upper- and lower-case letters (`hunter22x` and `wJalrXUtnFEMIK7MDENG` are; `it`, `None`, `/home/runner` are
    not)."""
    if name.lower().endswith("token") and not _STRONG_TOKEN_NAME.search(name):
        return _CREDENTIAL_VALUE.fullmatch(value) is not None
    if any(char.isdigit() for char in value):
        return True
    return len(value) >= 16 and value.lower() != value and value.upper() != value


def _redact_shape(match: re.Match) -> str:
    return (match.group(1) or "") + REDACTED


def _redact_assignment(match: re.Match) -> str:
    if not _key_like(match.group(1), match.group(3)):
        return match.group(0)
    return match.group(1) + match.group(2) + REDACTED


def _scrub_assignments(text: str) -> str:
    return SECRET_ASSIGNMENT.sub(_redact_assignment, text)


def without_known_bytes(content: bytes, known: Iterable[str]) -> bytes:
    """`content` with each known value (8 characters or more, longest first), encoded as UTF-8, replaced
    by REDACTED in the raw bytes. Exact values only: no key shapes (a file is evidence, and the shape
    rules rewrite prose) and no encoded forms. It works on any file, but a compressed one (xlsx, png,
    zip) keeps whatever its compression hides."""
    for value in _ordered_known(known):
        content = content.replace(value.encode(), REDACTED.encode())
    return content


def scrub_text(text: str, known: Iterable[str] = ()) -> str:
    """`text` with each known value (8 characters or more, longest first), each key shape and the key-like value of
    each secret assignment replaced by REDACTED."""
    for value in _ordered_known(known):
        text = text.replace(value, REDACTED)
    text = SECRET_SHAPE.sub(_redact_shape, text)
    return _scrub_assignments(text)
