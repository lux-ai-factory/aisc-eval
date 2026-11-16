"""Failure text cleared of secrets before the worker reports, raises or uploads it.

`scrub_text` replaces the values the worker knows it handed the plugin (`known_secrets`: the project's secrets,
the run's pass, the worker's own model keys, the secret-named values of the plugin configuration), then anything
shaped like a known kind of key, then the key-like value of a `password=...` style assignment.

The rule lives in one module shared by every Python service, aisc_redact (shared/identity), held to the shared
vectors redaction-cases.json; this module keeps the worker's names for it and adds what only the worker knows.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from aisc_redact import (  # noqa: F401 (the worker's names for the shared rule)
    CREDENTIAL,
    LEDGER_SECRET_NAME,
    MIN_KNOWN_LENGTH,
    REDACTED,
    SECRET_ASSIGNMENT,
    SECRET_SHAPE,
    _ordered_known,
    scrub_text,
    without_known_bytes,
)
from aisc_redact import secret_named_values as _named_values

#: The worker's own keys a plugin is handed.
WORKER_KEY_NAMES = ("AISC_TARGET_API_KEY", "API_KEY_OPENAI", "OPENAI_API_KEY")


def known_secrets(
    secret_env: Mapping[str, str],
    worker_env: Mapping[str, str],
    plugin_config,
    run_pass: str | None,
    also: Iterable[str] = (),
) -> list[str]:
    """The secret values a run hands its plugin (`also`: any other, such as its proxy token), 8 characters or
    more, longest first."""
    from aisc_eval.celery_tasks import (
        OPENAI_PLACEHOLDER,
    )  # the worker's module; imported late, it imports this

    values = list(secret_env.values())
    values.extend(also)
    if run_pass:
        values.append(run_pass)
    values.extend(
        worker_env[name]
        for name in WORKER_KEY_NAMES
        if worker_env.get(name) and worker_env[name] != OPENAI_PLACEHOLDER
    )
    _named_values(plugin_config, values)
    return _ordered_known(values)
